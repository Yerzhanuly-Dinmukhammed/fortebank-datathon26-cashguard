"""Benchmark frame-embedding models + a light point-supervised classifier.

Protocol: leave-one-video-out CV. Frames within +-POS_S of the labelled timecode are positives,
frames POS_S..IGNORE_S away are ignored, everything else (and all clean videos) are negatives.
Video score = max of the smoothed frame score; predicted timecode = its argmax.

Robustness proxy: the held-out video is re-embedded after strong geometric augmentation
(rotation, perspective, crop, flip, scale) and scored by the model trained on the other videos.

Usage: python research/embed_bench.py google/siglip2-base-patch16-224 [--aug-train]
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import CACHE_DIR, OUT_DIR, cached_frames, load_labels  # noqa: E402

POS_S, IGNORE_S = 1.5, 4.0
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"


def augment(frames, seed):
    """Strong geometric augmentation, same transform for every frame of a video."""
    import torchvision.transforms.v2.functional as F
    rng = np.random.default_rng(seed)
    x = torch.from_numpy(frames).permute(0, 3, 1, 2)
    h, w = x.shape[-2:]
    angle = float(rng.uniform(-25, 25))
    x = F.rotate(x, angle, expand=False)
    d = 0.18
    jitter = lambda: [int(rng.uniform(0, d) * w), int(rng.uniform(0, d) * h)]  # noqa: E731
    a, b, c, e = jitter(), jitter(), jitter(), jitter()
    start = [[0, 0], [w, 0], [w, h], [0, h]]
    end = [[a[0], a[1]], [w - b[0], b[1]], [w - c[0], h - c[1]], [e[0], h - e[1]]]
    x = F.perspective(x, start, end)
    s = float(rng.uniform(0.7, 0.9))
    ch, cw = int(h * s), int(w * s)
    top, left = int(rng.uniform(0, h - ch)), int(rng.uniform(0, w - cw))
    x = F.resized_crop(x, top, left, ch, cw, [h, w])
    if rng.random() < 0.5:
        x = F.horizontal_flip(x)
    return x.permute(0, 2, 3, 1).numpy(), {"angle": round(angle, 1), "crop": round(s, 2)}


class Embedder:
    def __init__(self, model_id):
        from transformers import AutoModel, AutoProcessor
        self.model_id = model_id
        self.proc = AutoProcessor.from_pretrained(model_id)
        self.model = AutoModel.from_pretrained(model_id, torch_dtype=torch.float16).to(DEVICE).eval()

    @torch.no_grad()
    def __call__(self, frames, bs=16):
        out = []
        for i in range(0, len(frames), bs):
            imgs = [Image.fromarray(f) for f in frames[i:i + bs]]
            inp = self.proc(images=imgs, return_tensors="pt").to(DEVICE)
            inp = {k: (v.half() if v.is_floating_point() else v) for k, v in inp.items()}
            if hasattr(self.model, "get_image_features"):
                e = self.model.get_image_features(**inp)
                e = getattr(e, "pooler_output", e)
            else:  # DINOv2-style: CLS token
                e = self.model(**inp).last_hidden_state[:, 0]
            out.append(torch.nn.functional.normalize(e.float(), dim=-1).cpu().numpy())
        return np.concatenate(out)


def features(emb):
    """Per-frame embedding + temporal context (delta to neighbours, local mean)."""
    prev = np.vstack([emb[:1], emb[:-1]])
    nxt = np.vstack([emb[1:], emb[-1:]])
    k = 3
    pad = np.pad(emb, ((k, k), (0, 0)), mode="edge")
    local = np.stack([pad[i:i + 2 * k + 1].mean(0) for i in range(len(emb))])
    return np.hstack([emb, emb - prev, nxt - emb, emb - local])


def frame_targets(times, t_sec):
    y = np.zeros(len(times))
    w = np.ones(len(times))
    if t_sec is not None:
        d = np.abs(times - t_sec)
        y[d <= POS_S] = 1
        w[(d > POS_S) & (d <= IGNORE_S)] = 0
    return y, w


def smooth(p, k=2):
    return np.convolve(np.pad(p, k, mode="edge"), np.ones(2 * k + 1) / (2 * k + 1), mode="valid")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model_id")
    ap.add_argument("--aug-train", action="store_true", help="also train on augmented copies")
    args = ap.parse_args()

    labels = load_labels()
    tag = args.model_id.replace("/", "__")
    emb_cache = CACHE_DIR / f"emb_{tag}.npz"
    t_load = time.perf_counter()
    embedder = Embedder(args.model_id)
    load_s = time.perf_counter() - t_load
    if DEVICE == "mps":
        torch.mps.empty_cache()

    data, timing = {}, []
    cached = dict(np.load(emb_cache, allow_pickle=True)) if emb_cache.exists() else {}
    for r in labels.itertuples():
        times, frames = cached_frames(r.video_id, r.path)
        t0 = time.perf_counter()
        key = r.video_id
        if key in cached:
            emb, emb_aug = cached[key], cached[key + "__aug"]
        else:
            emb = embedder(frames)
            aug_frames, _ = augment(frames, seed=hash(key) % 1000)
            emb_aug = embedder(aug_frames)
            cached[key], cached[key + "__aug"] = emb, emb_aug
            timing.append({"video_id": key, "n_frames": len(frames), "embed_s": (time.perf_counter() - t0) / 2})
        data[key] = (times, features(emb), features(emb_aug), r.violation, r.t_sec)
    np.savez(emb_cache, **cached)

    # ---- leave-one-video-out
    rows = []
    ids = list(data)
    for vid in ids:
        Xs, ys, ws = [], [], []
        for other in ids:
            if other == vid:
                continue
            times, F, Fa, viol, t = data[other]
            y, w = frame_targets(times, t if viol else None)
            Xs.append(F); ys.append(y); ws.append(w)
            if args.aug_train:
                Xs.append(Fa); ys.append(y); ws.append(w)
        X, y, w = np.vstack(Xs), np.concatenate(ys), np.concatenate(ws)
        keep = w > 0
        sc = StandardScaler().fit(X[keep])
        clf = LogisticRegression(C=0.05, class_weight="balanced", max_iter=3000).fit(sc.transform(X[keep]), y[keep])
        times, F, Fa, viol, t = data[vid]
        for variant, feats in (("clean", F), ("aug", Fa)):
            p = smooth(clf.predict_proba(sc.transform(feats))[:, 1])
            i = int(p.argmax())
            rows.append({"video_id": vid, "variant": variant, "violation": viol, "t_true": t,
                         "score": float(p[i]), "t_pred": float(times[i])})

    res = pd.DataFrame(rows)
    summary = {"model": args.model_id, "aug_train": args.aug_train, "device": DEVICE,
               "model_load_s": round(load_s, 1)}
    if timing:
        tm = pd.DataFrame(timing)
        summary.update(embed_s_p50=round(tm.embed_s.median(), 2), embed_s_max=round(tm.embed_s.max(), 2),
                       embed_ms_per_frame=round(1000 * (tm.embed_s / tm.n_frames).median(), 1))
    for variant, g in res.groupby("variant"):
        pos, neg = g[g.violation], g[~g.violation]
        # threshold = lowest score among true violations -> recall 1.0 by construction (optimistic);
        # report precision there, plus a fixed-threshold view from the clean CV.
        thr_r1 = pos.score.min()
        fp_at_r1 = int((neg.score >= thr_r1).sum())
        from sklearn.metrics import roc_auc_score
        auc = roc_auc_score(g.violation, g.score)
        err = (pos.t_pred - pos.t_true).abs()
        summary[variant] = {
            "video_auc": round(auc, 3),
            "fp_clean_videos_at_recall1": f"{fp_at_r1}/{len(neg)}",
            "precision_at_recall1": round(len(pos) / (len(pos) + fp_at_r1), 3),
            "t_err_median_s": round(err.median(), 2), "t_err_max_s": round(err.max(), 2),
            "t_within_2s": f"{int((err <= 2).sum())}/{len(pos)}",
        }
    # fixed threshold chosen on clean CV, applied to augmented (how much recall do we lose on a new angle?)
    clean = res[res.variant == "clean"]
    thr = clean[clean.violation].score.min()
    aug = res[res.variant == "aug"]
    summary["aug_recall_at_clean_threshold"] = round(float((aug[aug.violation].score >= thr).mean()), 3)
    summary["aug_fpr_at_clean_threshold"] = round(float((aug[~aug.violation].score >= thr).mean()), 3)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    name = f"embed_{tag}{'_augtrain' if args.aug_train else ''}"
    res.to_csv(OUT_DIR / f"{name}.csv", index=False)
    (OUT_DIR / f"{name}.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

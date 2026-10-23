"""Zero-shot VLM benchmark on sliding windows (mlx-vlm, Apple Silicon).

Each window = N frames spanning WIN seconds, sent as separate images with a short direct prompt.
Window score = confidence if the model says violation else 0; video score = max; timecode = best window.

Usage:
  python research/vlm_bench.py mlx-community/Qwen3.5-4B-MLX-4bit [--videos X-1,X-3] [--prompt full|no_entry]
                            [--stride 2] [--temp 0] [--seed 0] [--tag run1]
"""
import argparse
import json
import re
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import OUT_DIR, cached_frames, load_labels  # noqa: E402

WIN, N_FRAMES, SIDE = 4.0, 4, 384

TYPES_FULL = [
    "cash, a cash pack or a cash bag stays on the desk while the cashier is away from the desk or leaves the room",
    "the safe stays open while the cashier is away, or someone other than the cashier opens or reaches into the safe",
    "a person other than the cashier walks behind the cashier's desk or takes cash from the cashier's side",
    "the cashier uses a mobile phone or puts cash into a pocket, clothes or personal bag",
    "a cash bag or cash is handed over by the cashier without counting it",
]
NORMAL = [
    "a client enters, sits on the client side of the desk, gives or receives cash",
    "the cashier counts cash at the desk or uses the money counter",
    "the cashier puts cash into the safe, closes it and leaves with an empty desk",
]
PROMPTS = {
    "full": TYPES_FULL,
    # unknown-type test: the 'stranger behind the desk' type is removed on purpose
    "no_entry": TYPES_FULL[:2] + TYPES_FULL[3:],
}

TEMPLATE = """You see {n} consecutive frames (frame 1 is earliest, {span:.0f} seconds total) from an overhead security camera in a bank cash room. The cashier works at the desk with a money counter, a laptop and a grey safe.

NORMAL (not a violation):
{normal}

Does any of these visible violations happen in these frames?
{types}
- any other clearly visible breach of cash security

Only report what is clearly visible. Answer with JSON only:
{{"violation": true/false, "type": "<short type>", "frame": <1-{n}>, "confidence": <0-100>, "evidence": "<one sentence>"}}"""


def build_prompt(kind):
    types = "\n".join(f"- {t}" for t in PROMPTS[kind])
    normal = "\n".join(f"- {t}" for t in NORMAL)
    return TEMPLATE.format(n=N_FRAMES, span=WIN, types=types, normal=normal)


def parse(text):
    m = re.search(r"\{.*\}", text, re.S)
    try:
        d = json.loads(m.group(0)) if m else {}
    except json.JSONDecodeError:
        d = {}
    viol = d.get("violation")
    if isinstance(viol, str):
        viol = viol.lower() in ("true", "yes", "1")
    conf = d.get("confidence", 50)
    try:
        conf = float(conf)
    except (TypeError, ValueError):
        conf = 50.0
    return bool(viol), conf, d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model_id")
    ap.add_argument("--videos", default="")
    ap.add_argument("--prompt", default="full", choices=list(PROMPTS))
    ap.add_argument("--stride", type=float, default=2.0)
    ap.add_argument("--temp", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tag", default="")
    args = ap.parse_args()

    import mlx.core as mx
    from mlx_vlm import generate, load
    from mlx_vlm.prompt_utils import apply_chat_template

    mx.random.seed(args.seed)
    t0 = time.perf_counter()
    model, processor = load(args.model_id)
    load_s = time.perf_counter() - t0
    prompt_text = build_prompt(args.prompt)
    formatted = apply_chat_template(processor, model.config, prompt_text, num_images=N_FRAMES)

    labels = load_labels()
    if args.videos:
        keep = set(args.videos.split(","))
        labels = labels[labels.video_id.isin(keep)]

    tmp = Path(tempfile.mkdtemp())
    rows, wins = [], []
    for r in labels.itertuples():
        times, frames = cached_frames(r.video_id, r.path)
        tv = time.perf_counter()
        starts = np.arange(0, max(times[-1] - WIN, 0) + 1e-6, args.stride)
        best = (-1.0, 0.0, {})
        for s in starts:
            idx = np.linspace(np.searchsorted(times, s), np.searchsorted(times, s + WIN) - 1, N_FRAMES).round().astype(int)
            idx = np.clip(idx, 0, len(times) - 1)
            paths = []
            for k, i in enumerate(idx):
                img = Image.fromarray(frames[i])
                img.thumbnail((SIDE, SIDE))
                p = tmp / f"f{k}.jpg"
                img.save(p, quality=90)
                paths.append(str(p))
            out = generate(model, processor, formatted, image=paths, max_tokens=120,
                           temperature=args.temp, verbose=False)
            text = getattr(out, "text", out)
            viol, conf, d = parse(text)
            score = conf / 100 if viol else 0.0
            fr = d.get("frame", 2)
            try:
                t_evt = float(times[idx[min(max(int(fr), 1), N_FRAMES) - 1]])
            except (TypeError, ValueError):
                t_evt = float(times[idx[1]])
            wins.append({"video_id": r.video_id, "start": s, "score": score, "t_evt": t_evt,
                         "type": d.get("type", ""), "evidence": d.get("evidence", ""), "raw": text[:300]})
            if score > best[0]:
                best = (score, t_evt, d)
        dt = time.perf_counter() - tv
        rows.append({"video_id": r.video_id, "violation": r.violation, "t_true": r.t_sec,
                     "duration": float(times[-1]), "score": best[0], "t_pred": best[1],
                     "type": best[2].get("type", ""), "evidence": best[2].get("evidence", ""),
                     "n_windows": len(starts), "proc_s": round(dt, 1)})
        print(rows[-1], flush=True)

    res = pd.DataFrame(rows)
    pos, neg = res[res.violation], res[~res.violation]
    err = (pos.t_pred - pos.t_true).abs()
    from sklearn.metrics import roc_auc_score
    summary = {
        "model": args.model_id, "prompt": args.prompt, "temp": args.temp, "seed": args.seed,
        "n_videos": len(res), "model_load_s": round(load_s, 1),
        "proc_s_p50": float(res.proc_s.median()), "proc_s_max": float(res.proc_s.max()),
        "s_per_window": round(float(res.proc_s.sum() / res.n_windows.sum()), 2),
        "peak_mem_gb": round(mx.get_peak_memory() / 1e9, 2),
    }
    if len(pos) and len(neg):
        summary["video_auc"] = round(roc_auc_score(res.violation, res.score), 3)
    for thr in (0.5, 0.7):
        tp = int((pos.score >= thr).sum()); fp = int((neg.score >= thr).sum())
        summary[f"thr{thr}"] = {"recall": f"{tp}/{len(pos)}", "fp": f"{fp}/{len(neg)}",
                                "precision": round(tp / max(tp + fp, 1), 3)}
    if len(pos):
        summary.update(t_err_median_s=round(err.median(), 2), t_err_max_s=round(err.max(), 2),
                       t_within_2s=f"{int((err <= 2).sum())}/{len(pos)}")
    name = f"vlm_{args.model_id.replace('/', '__')}_{args.prompt}{'_' + args.tag if args.tag else ''}"
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    res.to_csv(OUT_DIR / f"{name}.csv", index=False)
    pd.DataFrame(wins).to_csv(OUT_DIR / f"{name}_windows.csv", index=False)
    (OUT_DIR / f"{name}.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

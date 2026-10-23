"""Compare open-vocabulary detectors on frames where cash / open safe are known to be visible.

Usage: python research/compare_ovd.py <dataset dir> <probe_frames.csv> <out.jpg> [device]
"""
import sys
import time
from pathlib import Path

import pandas as pd
import torch
from PIL import Image, ImageDraw

QUERIES = {
    "cash": ["banknotes", "a stack of paper money", "cash in hands", "a bundle of bills"],
    "safe_open": ["an open safe door", "an open metal safe"],
    "collection_bag": ["a green bag"],
    "handbag": ["a handbag"],
    "cup": ["a cup"],
    "phone": ["a mobile phone"],
}
COLORS = {"cash": "lime", "safe_open": "red", "collection_bag": "green", "handbag": "blue", "cup": "white", "phone": "yellow"}


def run_owl(model_id, imgs, device, thr=0.12):
    from transformers import Owlv2ForObjectDetection, Owlv2Processor
    proc = Owlv2Processor.from_pretrained(model_id)
    model = Owlv2ForObjectDetection.from_pretrained(model_id).to(device).eval()
    qs, qc = [], []
    for c, ql in QUERIES.items():
        qs += ql; qc += [c] * len(ql)
    out, t0 = [], time.time()
    for img in imgs:
        with torch.no_grad():
            inp = proc(text=[qs], images=img, return_tensors="pt").to(device)
            r = proc.post_process_grounded_object_detection(model(**inp), threshold=thr,
                                                             target_sizes=[(max(img.size),) * 2])[0]
        out.append([(qc[int(l)], float(s), [float(v) for v in b]) for b, l, s in
                    zip(r["boxes"].cpu(), r["labels"].cpu(), r["scores"].cpu())])
    return out, (time.time() - t0) / len(imgs)


def run_gdino(model_id, imgs, device, box_thr=0.3, text_thr=0.25):
    from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor
    proc = AutoProcessor.from_pretrained(model_id)
    model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id).to(device).eval()
    phrase2cls = {q: c for c, ql in QUERIES.items() for q in ql}
    text = " . ".join(phrase2cls) + " ."
    out, t0 = [], time.time()
    for img in imgs:
        with torch.no_grad():
            inp = proc(images=img, text=text, return_tensors="pt").to(device)
            r = proc.post_process_grounded_object_detection(model(**inp), inp.input_ids, threshold=box_thr,
                                                            text_threshold=text_thr, target_sizes=[img.size[::-1]])[0]
        dets = []
        for b, s, lab in zip(r["boxes"].cpu(), r["scores"].cpu(), r.get("text_labels", r.get("labels"))):
            lab = str(lab)
            cls = next((c for q, c in phrase2cls.items() if q in lab or lab in q), None)
            if cls:
                dets.append((cls, float(s), [float(v) for v in b]))
        out.append(dets)
    return out, (time.time() - t0) / len(imgs)


def nms(dets, thr=0.5):
    def iou(a, b):
        x0, y0, x1, y1 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
        i = max(0, x1 - x0) * max(0, y1 - y0)
        return i / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - i + 1e-6)
    keep = []
    for d in sorted(dets, key=lambda d: -d[1]):
        if not any(k[0] == d[0] and iou(k[2], d[2]) > thr for k in keep):
            keep.append(d)
    return keep


def main(ds, probe_csv, out, device="mps"):
    probe = pd.read_csv(probe_csv)
    imgs = [Image.open(Path(ds) / "dataset" / f).convert("RGB") for f in probe.file]
    results = {}
    for name, fn, mid in [("GroundingDINO-base", run_gdino, "IDEA-Research/grounding-dino-base"),
                          ("OWLv2-large", run_owl, "google/owlv2-large-patch14-ensemble")]:
        try:
            dets, sec = fn(mid, imgs, device)
        except Exception as e:  # MPS op gaps -> CPU
            print(name, "on", device, "failed:", type(e).__name__, str(e)[:120], "-> cpu")
            dets, sec = fn(mid, imgs, "cpu")
        results[name] = [nms(d) for d in dets]
        print(f"{name}: {sec:.2f} s/frame")
        for r, d in zip(probe.itertuples(), results[name]):
            found = sorted({c for c, s, b in d})
            print(f"   {r.v:5s} expect={r.expect:18s} found={found}")
    rows = []
    for name, dets in results.items():
        tiles = []
        for img, d, r in zip(imgs, dets, probe.itertuples()):
            im = img.copy(); dr = ImageDraw.Draw(im)
            for c, s, b in d:
                dr.rectangle(b, outline=COLORS[c], width=3); dr.text((b[0] + 2, b[1] + 2), f"{c} {s:.2f}", fill=COLORS[c])
            dr.text((4, 4), f"{name} | {r.v}: {r.expect}", fill="red")
            tiles.append(im.resize((im.width // 2, im.height // 2)))
        rows.append(tiles)
    w, h = rows[0][0].size
    sheet = Image.new("RGB", (w * len(rows[0]), h * len(rows)), "white")
    for i, tiles in enumerate(rows):
        for j, t in enumerate(tiles):
            sheet.paste(t, (j * w, i * h))
    sheet.save(out, quality=85)


if __name__ == "__main__":
    main(*sys.argv[1:])

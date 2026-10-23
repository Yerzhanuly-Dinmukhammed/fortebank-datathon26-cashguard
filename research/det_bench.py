"""Detector benchmark: speed + visual quality on labelled frames.

- RT-DETRv2 (COCO): person / cell phone / handbag / backpack boxes on every cached frame (2 fps).
- OWLv2 (open-vocabulary): text queries for cash, phone, bag, open safe on the labelled event frames.
Saves annotated frames to reports/bench/det/ for eyeballing.

Usage: python research/det_bench.py [--device cpu|mps] [--videos X-1,X-15]
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import OUT_DIR, cached_frames, load_labels  # noqa: E402

COCO_KEEP = {"person", "cell phone", "handbag", "backpack", "suitcase"}
OWL_QUERIES = ["a stack of banknotes", "a cash bag", "a mobile phone in a hand", "an open safe door", "a person"]


def draw(img, boxes, color):
    d = ImageDraw.Draw(img)
    for (x0, y0, x1, y1), label in boxes:
        d.rectangle([x0, y0, x1, y1], outline=color, width=2)
        d.text((x0 + 2, y0 + 2), label, fill=color)
    return img


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="mps")
    ap.add_argument("--videos", default="X-1,X-2,X-4,X-13,X-15,X-17,X-3,X-12")
    args = ap.parse_args()
    from transformers import (AutoImageProcessor, AutoModelForObjectDetection, Owlv2ForObjectDetection,
                              Owlv2Processor)

    dev = args.device
    out = OUT_DIR / "det"
    out.mkdir(parents=True, exist_ok=True)
    labels = load_labels()
    labels = labels[labels.video_id.isin(args.videos.split(","))]

    t0 = time.perf_counter()
    rt_proc = AutoImageProcessor.from_pretrained("PekingU/rtdetr_v2_r18vd")
    rt = AutoModelForObjectDetection.from_pretrained("PekingU/rtdetr_v2_r18vd").to(dev).eval()
    owl_proc = Owlv2Processor.from_pretrained("google/owlv2-base-patch16-ensemble")
    owl = Owlv2ForObjectDetection.from_pretrained("google/owlv2-base-patch16-ensemble").to(dev).eval()
    load_s = time.perf_counter() - t0

    summary = {"device": dev, "load_s": round(load_s, 1), "videos": {}}
    rt_times, owl_times = [], []
    for r in labels.itertuples():
        times, frames = cached_frames(r.video_id, r.path)
        counts = []
        t1 = time.perf_counter()
        with torch.no_grad():
            for i in range(0, len(frames), 8):
                imgs = [Image.fromarray(f) for f in frames[i:i + 8]]
                inp = rt_proc(images=imgs, return_tensors="pt").to(dev)
                res = rt_proc.post_process_object_detection(rt(**inp), threshold=0.5,
                                                            target_sizes=[im.size[::-1] for im in imgs])
                for rr in res:
                    names = [rt.config.id2label[int(l)] for l in rr["labels"]]
                    counts.append(sum(n == "person" for n in names))
        rt_times.append((time.perf_counter() - t1) / len(frames))
        t_evt = r.t_sec if r.violation else float(times[len(times) // 2])
        i_evt = int(np.abs(times - t_evt).argmin())
        img = Image.fromarray(frames[i_evt])
        # RT-DETR boxes on the event frame
        with torch.no_grad():
            inp = rt_proc(images=[img], return_tensors="pt").to(dev)
            rr = rt_proc.post_process_object_detection(rt(**inp), threshold=0.4, target_sizes=[img.size[::-1]])[0]
        rt_boxes = [(b.tolist(), f"{rt.config.id2label[int(l)]} {s:.2f}") for b, l, s in
                    zip(rr["boxes"].cpu(), rr["labels"].cpu(), rr["scores"].cpu())
                    if rt.config.id2label[int(l)] in COCO_KEEP]
        # OWLv2 on the same frame
        t2 = time.perf_counter()
        with torch.no_grad():
            inp = owl_proc(text=[OWL_QUERIES], images=img, return_tensors="pt").to(dev)
            o = owl(**inp)
            side = max(img.size)
            rr = owl_proc.post_process_grounded_object_detection(o, threshold=0.15, target_sizes=[(side, side)])[0]
        owl_times.append(time.perf_counter() - t2)
        owl_boxes = [(b.tolist(), f"{OWL_QUERIES[int(l)]} {s:.2f}") for b, l, s in
                     zip(rr["boxes"].cpu(), rr["labels"].cpu(), rr["scores"].cpu())]
        a = draw(img.copy(), rt_boxes, "red")
        b = draw(img.copy(), owl_boxes, "lime")
        sheet = Image.new("RGB", (a.width * 2, a.height))
        sheet.paste(a, (0, 0)); sheet.paste(b, (a.width, 0))
        sheet.save(out / f"{r.video_id}_t{t_evt:.0f}.jpg", quality=85)
        summary["videos"][r.video_id] = {
            "violation": bool(r.violation), "t": t_evt,
            "person_count_timeline": "".join(str(min(c, 9)) for c in counts),
            "rtdetr_event_frame": [lbl for _, lbl in rt_boxes],
            "owl_event_frame": [lbl for _, lbl in owl_boxes],
        }
    summary["rtdetr_ms_per_frame"] = round(1000 * float(np.median(rt_times)), 1)
    summary["owlv2_ms_per_frame"] = round(1000 * float(np.median(owl_times)), 1)
    (OUT_DIR / f"det_{dev}.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

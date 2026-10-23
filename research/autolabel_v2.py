"""Auto-labels v2 for all detector frames (stage A: detectors, run with the system python that has ultralytics).

- YOLO11m (COCO): phone, cup, handbag
- OWLv2-large: cash, collection_bag, safe (state decided later by Qwen), counting_machine
Raw detections -> <out>/raw_dets.json; stage B (autolabel_v2_safe.py) classifies safe boxes as open/closed
and writes the final COCO zips.

Usage: python research/autolabel_v2.py <dataset dir> <out dir> [device]
"""
import json
import sys
import time
from pathlib import Path

import pandas as pd
import torch
from PIL import Image

YOLO_MAP = {"cell phone": "phone", "cup": "cup", "wine glass": "cup", "handbag": "handbag"}
OWL = {  # class: (queries, threshold)
    "cash": (["banknotes", "a stack of paper money", "cash in hands", "a bundle of bills"], 0.15),
    "collection_bag": (["a green cash collection bag", "a green bag"], 0.15),
    "safe": (["a closed grey metal safe", "a small grey steel safe standing on a white cabinet", "an open safe door"], 0.15),
    "counting_machine": (["a money counting machine", "a banknote counter machine"], 0.15),
}


def main(ds, out, device="mps"):
    from transformers import Owlv2ForObjectDetection, Owlv2Processor
    from ultralytics import YOLO
    ds, out = Path(ds), Path(out)
    out.mkdir(parents=True, exist_ok=True)
    manifest = pd.read_csv(ds / "dataset" / "manifest.csv")
    yolo = YOLO(str(Path(__file__).resolve().parents[1] / "models" / "yolo11m.pt"))
    mid = "google/owlv2-large-patch14-ensemble"
    proc = Owlv2Processor.from_pretrained(mid)
    owl = Owlv2ForObjectDetection.from_pretrained(mid).to(device).eval()
    queries, qcls = [], []
    for c, (qs, _) in OWL.items():
        queries += qs; qcls += [c] * len(qs)
    raw_path = out / "raw_dets.json"
    raw = json.loads(raw_path.read_text()) if raw_path.exists() else {}
    t0 = time.time()
    for k, f in enumerate(manifest.file):
        if f in raw:
            continue
        img = Image.open(ds / "dataset" / f).convert("RGB")
        dets = []
        r = yolo.predict(img, conf=0.25, verbose=False, device=device)[0]
        for b, c, s in zip(r.boxes.xyxy.cpu().numpy(), r.boxes.cls.cpu().numpy(), r.boxes.conf.cpu().numpy()):
            if r.names[int(c)] in YOLO_MAP:
                dets.append([YOLO_MAP[r.names[int(c)]], float(s), [float(v) for v in b]])
        with torch.no_grad():
            inp = proc(text=[queries], images=img, return_tensors="pt").to(device)
            res = proc.post_process_grounded_object_detection(owl(**inp), threshold=0.1,
                                                              target_sizes=[(max(img.size),) * 2])[0]
        for b, l, s in zip(res["boxes"].cpu().numpy(), res["labels"].cpu().numpy(), res["scores"].cpu().numpy()):
            c = qcls[int(l)]
            if s >= OWL[c][1]:
                x0, y0, x1, y1 = [float(v) for v in b]
                dets.append([c, float(s), [max(0, x0), max(0, y0), min(img.width, x1), min(img.height, y1)]])
        raw[f] = {"w": img.width, "h": img.height, "split": manifest.split[k], "dets": dets}
        if (k + 1) % 25 == 0:
            raw_path.write_text(json.dumps(raw))
            el = time.time() - t0
            print(f"{k + 1}/{len(manifest)}  {el / 60:.1f} min", flush=True)
    raw_path.write_text(json.dumps(raw))
    print("done", len(raw), f"{(time.time() - t0) / 60:.1f} min", flush=True)


if __name__ == "__main__":
    main(*sys.argv[1:])

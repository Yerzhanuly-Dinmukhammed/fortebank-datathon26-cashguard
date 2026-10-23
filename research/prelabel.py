"""Draft bounding boxes for CVAT: YOLO11m (COCO) + OWLv2 (open vocabulary) -> COCO 1.0 JSON.

YOLO11m (COCO weights):  phone (cell phone), cup (cup, wine glass), handbag
OWLv2 (text queries):    cash, collection_bag, safe_open, safe_closed, counting_machine, id_document

Output per subset: <out>/<subset>/annotations/instances_default.json (+ zip for CVAT "COCO 1.0" import),
a scores CSV and a preview sheet. Boxes are drafts: annotators must review every frame.

Usage:
  python research/prelabel.py --dataset <dir with dataset/manifest.csv> --out data/processed/prelabels \
      --subset valid --subset train_p1 [--limit 5]
"""
import argparse
import json
import time
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageDraw

CATEGORIES = ["cash", "collection_bag", "safe_open", "safe_closed", "counting_machine",
              "id_document", "phone", "handbag", "cup"]
CAT_ID = {c: i + 1 for i, c in enumerate(CATEGORIES)}

YOLO_MAP = {"cell phone": "phone", "cup": "cup", "wine glass": "cup", "handbag": "handbag"}
YOLO_CONF = 0.25

# several phrasings per class; OWLv2 score threshold per class (scores are low by nature)
OWL_QUERIES = {
    "cash": (["a stack of banknotes", "paper money bills", "a fan of banknotes in a hand", "a bundle of cash"], 0.14),
    "collection_bag": (["a green cash collection bag", "a green sealed bag"], 0.14),
    # plain "grey metal safe" misses the safe entirely; this phrasing finds it in layouts A/D/E,
    # in layout B a white cabinet ranks first -> keep 2 candidates, annotators delete the wrong one
    "safe_open": (["an open safe with its door open", "a small grey steel safe with an open door"], 0.2),
    "safe_closed": (["a small grey steel safe standing on a white cabinet"], 0.2),
    "counting_machine": (["a money counting machine", "a banknote counter machine"], 0.14),
    "id_document": (["an ID card", "a plastic bank card", "a passport"], 0.2),
}
MAX_PER_CLASS = {"safe_open": 1, "safe_closed": 2, "counting_machine": 2, "collection_bag": 2}

COLORS = {"cash": "lime", "collection_bag": "green", "safe_open": "red", "safe_closed": "orange",
          "counting_machine": "cyan", "id_document": "magenta", "phone": "yellow", "handbag": "blue", "cup": "white"}


def iou(a, b):
    x0, y0 = max(a[0], b[0]), max(a[1], b[1])
    x1, y1 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, x1 - x0) * max(0, y1 - y0)
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def nms(dets, thr=0.5):
    """dets: list of (cls, score, box). Class-wise NMS + per-class caps."""
    out = []
    for d in sorted(dets, key=lambda d: -d[1]):
        if any(o[0] == d[0] and iou(o[2], d[2]) > thr for o in out):
            continue
        if sum(o[0] == d[0] for o in out) >= MAX_PER_CLASS.get(d[0], 99):
            continue
        out.append(d)
    return out


def resolve_conflicts(dets):
    """Same safe can't be both open and closed; a counting machine is not a handbag."""
    keep = []
    for d in sorted(dets, key=lambda d: -d[1]):
        if d[0] in ("safe_open", "safe_closed") and any(
                k[0] in ("safe_open", "safe_closed") and iou(k[2], d[2]) > 0.3 for k in keep):
            continue
        if d[0] == "handbag" and any(k[0] == "counting_machine" and iou(k[2], d[2]) > 0.3 for k in dets):
            continue
        keep.append(d)
    return keep


class Labeler:
    def __init__(self, device, owl_id):
        from transformers import Owlv2ForObjectDetection, Owlv2Processor
        from ultralytics import YOLO
        self.device = device
        self.yolo = YOLO("yolo11m.pt")
        self.owl_proc = Owlv2Processor.from_pretrained(owl_id)
        self.owl = Owlv2ForObjectDetection.from_pretrained(owl_id).to(device).eval()
        self.queries, self.q_cls = [], []
        for cls, (qs, _) in OWL_QUERIES.items():
            self.queries += qs
            self.q_cls += [cls] * len(qs)

    def yolo_dets(self, img):
        r = self.yolo.predict(img, conf=YOLO_CONF, verbose=False, device=self.device)[0]
        out = []
        for b, c, s in zip(r.boxes.xyxy.cpu().numpy(), r.boxes.cls.cpu().numpy(), r.boxes.conf.cpu().numpy()):
            name = r.names[int(c)]
            if name in YOLO_MAP:
                out.append((YOLO_MAP[name], float(s), [float(v) for v in b]))
        return out

    @torch.no_grad()
    def owl_dets(self, img):
        inp = self.owl_proc(text=[self.queries], images=img, return_tensors="pt").to(self.device)
        o = self.owl(**inp)
        side = max(img.size)  # OWLv2 pads to a square at the bottom/right
        res = self.owl_proc.post_process_grounded_object_detection(o, threshold=0.1, target_sizes=[(side, side)])[0]
        out = []
        for b, l, s in zip(res["boxes"].cpu().numpy(), res["labels"].cpu().numpy(), res["scores"].cpu().numpy()):
            cls = self.q_cls[int(l)]
            if s < OWL_QUERIES[cls][1]:
                continue
            x0, y0, x1, y1 = [float(v) for v in b]
            x0, y0 = max(0, x0), max(0, y0)
            x1, y1 = min(img.width, x1), min(img.height, y1)
            if x1 - x0 < 4 or y1 - y0 < 4:
                continue
            out.append((cls, float(s), [x0, y0, x1, y1]))
        return out

    def __call__(self, img):
        return resolve_conflicts(nms(self.yolo_dets(img) + self.owl_dets(img)))


def select(manifest, subset):
    if subset == "valid":
        return manifest[manifest.split == "valid"]
    if subset.startswith("train_p"):
        return manifest[(manifest.split == "train") & (manifest.label_priority == int(subset[-1]))]
    raise ValueError(subset)


def write_coco(rows, images_meta, path):
    coco = {"info": {"description": "DATATHON26 draft boxes (YOLO11m COCO + OWLv2), review required"},
            "licenses": [], "categories": [{"id": CAT_ID[c], "name": c, "supercategory": ""} for c in CATEGORIES],
            "images": [], "annotations": []}
    for img_id, (fname, w, h) in enumerate(images_meta, start=1):
        coco["images"].append({"id": img_id, "file_name": fname, "width": w, "height": h})
    name_to_id = {m[0]: i for i, m in enumerate(images_meta, start=1)}
    for ann_id, r in enumerate(rows, start=1):
        x0, y0, x1, y1 = r["box"]
        coco["annotations"].append({"id": ann_id, "image_id": name_to_id[r["file"]], "category_id": CAT_ID[r["cls"]],
                                    "bbox": [round(x0, 1), round(y0, 1), round(x1 - x0, 1), round(y1 - y0, 1)],
                                    "area": round((x1 - x0) * (y1 - y0), 1), "iscrowd": 0, "segmentation": [],
                                    "attributes": {"occluded": False}})
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(coco, ensure_ascii=False))


def preview(samples, path, cols=6):
    tiles = []
    for img, dets, fname in samples:
        im = img.copy()
        d = ImageDraw.Draw(im)
        for cls, s, b in dets:
            d.rectangle(b, outline=COLORS[cls], width=3)
            d.text((b[0] + 3, b[1] + 3), f"{cls} {s:.2f}", fill=COLORS[cls])
        d.text((4, 4), fname, fill="red")
        tiles.append(im.resize((im.width // 2, im.height // 2)))
    w, h = tiles[0].size
    sheet = Image.new("RGB", (cols * w, int(np.ceil(len(tiles) / cols)) * h), "white")
    for k, t in enumerate(tiles):
        sheet.paste(t, ((k % cols) * w, (k // cols) * h))
    sheet.save(path, quality=85)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--subset", action="append", required=True)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--owl", default="google/owlv2-base-patch16-ensemble")
    ap.add_argument("--device", default="mps" if torch.backends.mps.is_available() else "cpu")
    args = ap.parse_args()

    root = Path(args.dataset)
    manifest = pd.read_csv(root / "dataset" / "manifest.csv")
    t0 = time.perf_counter()
    labeler = Labeler(args.device, args.owl)
    print(f"models loaded in {time.perf_counter() - t0:.1f}s on {args.device}", flush=True)

    for subset in args.subset:
        sel = select(manifest, subset)
        if args.limit:
            sel = sel.head(args.limit)
        rows, meta, samples, times = [], [], [], []
        for k, f in enumerate(sel.file):
            img = Image.open(root / "dataset" / f).convert("RGB")
            fname = Path(f).name
            t1 = time.perf_counter()
            dets = labeler(img)
            times.append(time.perf_counter() - t1)
            meta.append((fname, img.width, img.height))
            rows += [{"file": fname, "cls": c, "score": round(s, 3), "box": b} for c, s, b in dets]
            if k % max(1, len(sel) // 24) == 0 and len(samples) < 24:
                samples.append((img, dets, fname))
            if (k + 1) % 25 == 0:
                print(f"  {subset}: {k + 1}/{len(sel)}", flush=True)
        out = Path(args.out) / subset
        json_path = out / "annotations" / "instances_default.json"
        write_coco(rows, meta, json_path)
        with zipfile.ZipFile(out / f"{subset}_coco1.0.zip", "w", zipfile.ZIP_DEFLATED) as z:
            z.write(json_path, "annotations/instances_default.json")
        df = pd.DataFrame([{**r, "box": [round(v, 1) for v in r["box"]]} for r in rows])
        df.to_csv(out / "draft_scores.csv", index=False)
        preview(samples, out / "preview.jpg")
        counts = df.cls.value_counts().reindex(CATEGORIES, fill_value=0).to_dict() if len(df) else {}
        print(json.dumps({"subset": subset, "images": len(meta), "boxes": len(rows),
                          "images_without_boxes": len(meta) - (df.file.nunique() if len(df) else 0),
                          "per_class": counts, "s_per_image": round(float(np.median(times)), 2)},
                         ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()

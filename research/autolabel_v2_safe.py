"""Auto-labels v2, stage B (run with .venv python that has mlx-vlm):
clean raw detections, decide safe open/closed with Qwen3.5-4B on safe crops, write COCO zips.

Usage: .venv/bin/python research/autolabel_v2_safe.py <dataset dir> <autolabels dir>
"""
import json
import sys
import tempfile
import zipfile
from collections import Counter
from pathlib import Path

from PIL import Image, ImageDraw

CLASSES = ["cash", "collection_bag", "safe_open", "safe_closed", "counting_machine",
           "id_document", "phone", "handbag", "cup"]
CAP = {"counting_machine": 2, "collection_bag": 2, "safe": 2, "phone": 2, "handbag": 2, "cup": 2}
QWEN_Q = ("This is a crop from a security camera showing a grey metal safe (or a white cabinet). Is the door of the "
          "grey metal safe open, so that the inside of the safe or the swung-out door is visible? Answer only yes or no.")


def iou(a, b):
    x0, y0, x1, y1 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    i = max(0, x1 - x0) * max(0, y1 - y0)
    return i / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - i + 1e-6)


def nms(dets, thr=0.5):
    keep = []
    for d in sorted(dets, key=lambda d: -d[1]):
        if any(k[0] == d[0] and iou(k[2], d[2]) > thr for k in keep):
            continue
        if sum(k[0] == d[0] for k in keep) >= CAP.get(d[0], 99):
            continue
        keep.append(d)
    return keep


def drop_stand(safes):
    """Two stacked boxes (safe + white stand under it): keep the upper one."""
    if len(safes) < 2:
        return safes
    a, b = sorted(safes, key=lambda d: d[2][1])[:2]
    xo = max(0, min(a[2][2], b[2][2]) - max(a[2][0], b[2][0])) / max(1, min(a[2][2] - a[2][0], b[2][2] - b[2][0]))
    return [a] if xo > 0.5 and b[2][1] > a[2][1] + 0.3 * (a[2][3] - a[2][1]) else safes


def clean(rec):
    W, H = rec["w"], rec["h"]
    out = []
    for c, s, b in rec["dets"]:
        cx, cy, area = (b[0] + b[2]) / 2, (b[1] + b[3]) / 2, (b[2] - b[0]) * (b[3] - b[1])
        if c == "cash" and (cy < 0.30 * H or area > 0.03 * W * H):   # glass wall / huge boxes are not cash
            continue
        out.append((c, s, b))
    out = nms(out)
    safes = drop_stand([d for d in out if d[0] == "safe"])
    others = [d for d in out if d[0] != "safe"]
    machines = [d for d in others if d[0] == "counting_machine"]
    others = [d for d in others if not (d[0] == "handbag" and any(iou(d[2], m[2]) > 0.3 for m in machines))]
    return others, safes


def main(ds, adir):
    from mlx_vlm import generate, load
    from mlx_vlm.prompt_utils import apply_chat_template
    ds, adir = Path(ds), Path(adir)
    raw = json.loads((adir / "raw_dets.json").read_text())
    model, proc = load("mlx-community/Qwen3.5-4B-MLX-4bit")
    prompt = apply_chat_template(proc, model.config, QWEN_Q, num_images=1)
    tmp = Path(tempfile.mkdtemp())
    final, stats = {}, Counter()
    for k, (f, rec) in enumerate(raw.items()):
        others, safes = clean(rec)
        img = Image.open(ds / "dataset" / f).convert("RGB")
        labelled = list(others)
        for j, (c, s, b) in enumerate(safes):
            w, h = b[2] - b[0], b[3] - b[1]
            crop = img.crop((max(0, b[0] - .25 * w), max(0, b[1] - .25 * h), min(img.width, b[2] + .25 * w),
                             min(img.height, b[3] + .25 * h)))
            crop = crop.resize((crop.width * 2, crop.height * 2)); p = tmp / "s.jpg"; crop.save(p)
            ans = getattr(generate(model, proc, prompt, image=[str(p)], max_tokens=3, temperature=0.0,
                                   verbose=False), "text", "").strip().lower()
            labelled.append(("safe_open" if ans.startswith("yes") else "safe_closed", s, b))
        final[f] = {**rec, "labels": labelled}
        stats.update(c for c, _, _ in labelled)
        if (k + 1) % 100 == 0:
            print(f"{k + 1}/{len(raw)}", dict(stats), flush=True)
    for split in ("train", "valid"):
        coco = {"images": [], "annotations": [], "categories": [{"id": i + 1, "name": c} for i, c in enumerate(CLASSES)]}
        for f, rec in final.items():
            if rec["split"] != split:
                continue
            img_id = len(coco["images"]) + 1
            coco["images"].append({"id": img_id, "file_name": Path(f).name, "width": rec["w"], "height": rec["h"]})
            for c, s, b in rec["labels"]:
                coco["annotations"].append({"id": len(coco["annotations"]) + 1, "image_id": img_id,
                                            "category_id": CLASSES.index(c) + 1,
                                            "bbox": [b[0], b[1], b[2] - b[0], b[3] - b[1]], "score": round(s, 3)})
        with zipfile.ZipFile(adir / f"{split}_auto.zip", "w", zipfile.ZIP_DEFLATED) as z:
            z.writestr("annotations/instances_default.json", json.dumps(coco, ensure_ascii=False))
        print(split, len(coco["images"]), "images", len(coco["annotations"]), "boxes")
    print("per class:", dict(stats))
    # preview: 24 frames with cash or open safe
    pick = [f for f, r in final.items() if any(c in ("cash", "safe_open") for c, _, _ in r["labels"])][:24]
    tiles = []
    col = {"cash": "lime", "safe_open": "red", "safe_closed": "orange", "collection_bag": "green",
           "counting_machine": "cyan", "phone": "yellow", "handbag": "blue", "cup": "white"}
    for f in pick:
        im = Image.open(ds / "dataset" / f).convert("RGB"); d = ImageDraw.Draw(im)
        for c, s, b in final[f]["labels"]:
            d.rectangle(b, outline=col.get(c, "white"), width=3); d.text((b[0] + 2, b[1] + 2), c, fill=col.get(c, "white"))
        d.text((4, 4), Path(f).name, fill="red")
        tiles.append(im.resize((im.width // 2, im.height // 2)))
    if tiles:
        w, h = tiles[0].size
        sheet = Image.new("RGB", (6 * w, ((len(tiles) + 5) // 6) * h), "white")
        for i, t in enumerate(tiles):
            sheet.paste(t, ((i % 6) * w, (i // 6) * h))
        sheet.save(adir / "preview.jpg", quality=85)


if __name__ == "__main__":
    main(*sys.argv[1:])

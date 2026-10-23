"""Build Label Studio import files (images embedded as base64 + draft boxes as predictions).

Images are embedded so no storage configuration is needed on the server; `fname` keeps the
original file name for the conversion back to COCO (src/ls_convert.py).

Usage: python research/make_ls_tasks.py <dataset dir with dataset/images> <draft COCO zip> <out.json>
"""
import base64
import json
import sys
import unicodedata
import zipfile
from collections import defaultdict
from pathlib import Path

CLASSES = ["cash", "collection_bag", "safe_open", "safe_closed", "counting_machine",
           "id_document", "phone", "handbag", "cup"]

CONFIG = """<View>
  <Header value="$fname"/>
  <Image name="image" value="$image" zoom="true" zoomControl="true" rotateControl="false"/>
  <RectangleLabels name="label" toName="image" strokeWidth="2">
""" + "".join(f'    <Label value="{c}" hotkey="{i + 1}"/>\n' for i, c in enumerate(CLASSES)) + """  </RectangleLabels>
</View>
"""


def nfc(s):
    return unicodedata.normalize("NFC", s)


def main(ds_dir, coco_zip, out_path):
    ds = Path(ds_dir) / "dataset" / "images"
    index = {nfc(p.name): p for p in ds.rglob("*.jpg")}
    with zipfile.ZipFile(coco_zip) as z:
        coco = json.loads(z.read("annotations/instances_default.json"))
    cats = {c["id"]: c["name"] for c in coco["categories"]}
    anns = defaultdict(list)
    for a in coco["annotations"]:
        anns[a["image_id"]].append(a)
    tasks = []
    for im in coco["images"]:
        name = nfc(im["file_name"])
        W, H = im["width"], im["height"]
        b64 = base64.b64encode(index[name].read_bytes()).decode()
        result = []
        for k, a in enumerate(anns[im["id"]]):
            x, y, w, h = a["bbox"]
            result.append({"id": f"{im['id']}_{k}", "from_name": "label", "to_name": "image", "type": "rectanglelabels",
                           "original_width": W, "original_height": H, "image_rotation": 0,
                           "value": {"x": 100 * x / W, "y": 100 * y / H, "width": 100 * w / W, "height": 100 * h / H,
                                     "rotation": 0, "rectanglelabels": [cats[a["category_id"]]]}})
        tasks.append({"data": {"image": f"data:image/jpeg;base64,{b64}", "fname": name, "width": W, "height": H},
                      "predictions": [{"model_version": "draft_yolo_owl", "result": result}]})
    Path(out_path).write_text(json.dumps(tasks, ensure_ascii=False))
    print(out_path, len(tasks), "tasks", round(Path(out_path).stat().st_size / 1e6, 1), "MB")


if __name__ == "__main__":
    if sys.argv[1] == "--config":
        Path(sys.argv[2]).write_text(CONFIG)
        print("config ->", sys.argv[2])
    else:
        main(*sys.argv[1:4])

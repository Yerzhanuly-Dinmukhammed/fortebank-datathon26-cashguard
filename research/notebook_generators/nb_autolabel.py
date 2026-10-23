import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from nb_runpod import nb  # noqa: E402

CELLS = [
    ("md", """
# 05 — Авторазметка v2 на GPU пода

Черновик v1 (OWLv2-base) почти не нашёл деньги и ни разу — открытый сейф. v2:
- **OWLv2-large** — деньги, инкассаторская сумка, сейф, счётная машинка (на наших кадрах деньги находит в 6 из 8 проверенных);
- **YOLO11m (COCO)** — телефон, стакан, личная сумка;
- **Qwen3.5-4B** — по вырезке каждого сейфа решает: открыт или закрыт (на проверке: закрытые 12/12, открытый X-20 — 3/3);
- чистка: без `id_document` (в v1 почти всё было ложным), без белой подставки под сейфом, без «денег» в зоне стеклянной стены.

Результат: `/workspace/data/autolabels/train_auto.zip`, `valid_auto.zip` → дальше `03` с `MODE = "auto"`.
"""),
    ("code", r'''
import os, sys, json, time, subprocess, zipfile, tempfile
from pathlib import Path
from collections import Counter

WORK = Path(os.getenv("WORK", "/workspace"))
DEVICE = os.getenv("DEVICE", "cuda")
os.environ.setdefault("HF_HOME", str(WORK / "hf_cache"))     # модели с HF сохраняются на volume
DS = WORK / "data" / "detds"
OUT = WORK / "data" / "autolabels"; OUT.mkdir(parents=True, exist_ok=True)
os.chdir(WORK)

r = subprocess.run([sys.executable, "-m", "pip", "install", "-q", "transformers>=5.2", "accelerate", "ultralytics==8.4.180",
                    "pandas", "pillow"], capture_output=True, text=True)
print(r.stdout[-1500:], r.stderr[-1500:]); assert r.returncode == 0

import pandas as pd, torch
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
print("torch", torch.__version__, "| GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "нет")
manifest = pd.read_csv(DS / "dataset" / "manifest.csv")
print("кадров:", len(manifest), manifest.split.value_counts().to_dict())
'''),
    ("md", "## 1. Детекторы: YOLO11m + OWLv2-large"),
    ("code", r'''
from transformers import Owlv2ForObjectDetection, Owlv2Processor
from ultralytics import YOLO

YOLO_MAP = {"cell phone": "phone", "cup": "cup", "wine glass": "cup", "handbag": "handbag"}
OWL = {  # класс: (запросы, порог)
    "cash": (["banknotes", "a stack of paper money", "cash in hands", "a bundle of bills"], 0.15),
    "collection_bag": (["a green cash collection bag", "a green bag"], 0.15),
    "safe": (["a closed grey metal safe", "a small grey steel safe standing on a white cabinet", "an open safe door"], 0.15),
    "counting_machine": (["a money counting machine", "a banknote counter machine"], 0.15),
}
yolo = YOLO("yolo11m.pt")
owl_id = "google/owlv2-large-patch14-ensemble"
owl_proc = Owlv2Processor.from_pretrained(owl_id)
owl = Owlv2ForObjectDetection.from_pretrained(owl_id, torch_dtype=torch.float16).to(DEVICE).eval()
queries, qcls = [], []
for c, (qs, _) in OWL.items():
    queries += qs; qcls += [c] * len(qs)

raw, t0 = {}, time.time()
for k, f in enumerate(manifest.file):
    img = Image.open(DS / "dataset" / f).convert("RGB")
    dets = []
    r = yolo.predict(img, conf=0.25, verbose=False, device=0 if DEVICE == "cuda" else DEVICE)[0]
    for b, c, s in zip(r.boxes.xyxy.cpu().numpy(), r.boxes.cls.cpu().numpy(), r.boxes.conf.cpu().numpy()):
        if r.names[int(c)] in YOLO_MAP:
            dets.append([YOLO_MAP[r.names[int(c)]], float(s), [float(v) for v in b]])
    with torch.no_grad():
        inp = owl_proc(text=[queries], images=img, return_tensors="pt").to(DEVICE)
        inp["pixel_values"] = inp["pixel_values"].half()
        res = owl_proc.post_process_grounded_object_detection(owl(**inp), threshold=0.1,
                                                              target_sizes=[(max(img.size),) * 2])[0]
    for b, l, s in zip(res["boxes"].float().cpu().numpy(), res["labels"].cpu().numpy(), res["scores"].float().cpu().numpy()):
        c = qcls[int(l)]
        if s >= OWL[c][1]:
            x0, y0, x1, y1 = [float(v) for v in b]
            dets.append([c, float(s), [max(0, x0), max(0, y0), min(img.width, x1), min(img.height, y1)]])
    raw[f] = {"w": img.width, "h": img.height, "split": manifest.split[k], "dets": dets}
    if (k + 1) % 100 == 0:
        print(f"{k + 1}/{len(manifest)}  {(time.time() - t0) / 60:.1f} мин", flush=True)
(OUT / "raw_dets.json").write_text(json.dumps(raw))
print(f"готово за {(time.time() - t0) / 60:.1f} мин")
del owl; torch.cuda.empty_cache()
'''),
    ("md", "## 2. Чистка + Qwen: сейф открыт или закрыт"),
    ("code", r'''
CLASSES = ["cash", "collection_bag", "safe_open", "safe_closed", "counting_machine",
           "id_document", "phone", "handbag", "cup"]
CAP = {"counting_machine": 2, "collection_bag": 2, "safe": 2, "phone": 2, "handbag": 2, "cup": 2}

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
    """Сейф + белая подставка под ним (две рамки одна над другой) -> оставляем верхнюю."""
    if len(safes) < 2:
        return safes
    a, b = sorted(safes, key=lambda d: d[2][1])[:2]
    xo = max(0, min(a[2][2], b[2][2]) - max(a[2][0], b[2][0])) / max(1, min(a[2][2] - a[2][0], b[2][2] - b[2][0]))
    return [a] if xo > 0.5 and b[2][1] > a[2][1] + 0.3 * (a[2][3] - a[2][1]) else safes

def clean(rec):
    W, H = rec["w"], rec["h"]
    out = []
    for c, s, b in rec["dets"]:
        cy, area = (b[1] + b[3]) / 2, (b[2] - b[0]) * (b[3] - b[1])
        if c == "cash" and (cy < 0.30 * H or area > 0.03 * W * H):   # стеклянная стена / огромные рамки — не деньги
            continue
        out.append((c, s, b))
    out = nms(out)
    safes = drop_stand([d for d in out if d[0] == "safe"])
    others = [d for d in out if d[0] != "safe"]
    machines = [d for d in others if d[0] == "counting_machine"]
    others = [d for d in others if not (d[0] == "handbag" and any(iou(d[2], m[2]) > 0.3 for m in machines))]
    return others, safes
'''),
    ("code", r'''
from transformers import AutoModelForImageTextToText, AutoProcessor

qwen_id = "Qwen/Qwen3.5-4B"
q_proc = AutoProcessor.from_pretrained(qwen_id)
qwen = AutoModelForImageTextToText.from_pretrained(qwen_id, torch_dtype=torch.bfloat16).to(DEVICE).eval()
QWEN_Q = ("This is a crop from a security camera showing a grey metal safe (or a white cabinet). Is the door of the "
          "grey metal safe open, so that the inside of the safe or the swung-out door is visible? Answer only yes or no.")

@torch.no_grad()
def safe_is_open(crop):
    msgs = [{"role": "user", "content": [{"type": "image", "image": crop}, {"type": "text", "text": QWEN_Q}]}]
    try:
        inp = q_proc.apply_chat_template(msgs, add_generation_prompt=True, tokenize=True, return_dict=True,
                                         return_tensors="pt", enable_thinking=False)
    except TypeError:
        inp = q_proc.apply_chat_template(msgs, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt")
    inp = inp.to(DEVICE)
    out = qwen.generate(**inp, max_new_tokens=8, do_sample=False)
    text = q_proc.batch_decode(out[:, inp["input_ids"].shape[1]:], skip_special_tokens=True)[0].strip().lower()
    return text.replace("<think>", "").replace("</think>", "").strip().startswith("yes"), text

# быстрая проверка на известных кадрах: X-20 (сейф открыт) и X-4 (закрыт)
for f in [x for x in manifest.file if "-20_t0004" in x or "-4_t0055" in x][:2]:
    _, safes = clean(raw[f]); img = Image.open(DS / "dataset" / f).convert("RGB")
    for c, s, b in safes:
        w, h = b[2] - b[0], b[3] - b[1]
        crop = img.crop((max(0, b[0] - .25 * w), max(0, b[1] - .25 * h), min(img.width, b[2] + .25 * w), min(img.height, b[3] + .25 * h)))
        print(Path(f).name, "->", safe_is_open(crop.resize((crop.width * 2, crop.height * 2))))
'''),
    ("code", r'''
final, stats, t0 = {}, Counter(), time.time()
for k, (f, rec) in enumerate(raw.items()):
    others, safes = clean(rec)
    img = Image.open(DS / "dataset" / f).convert("RGB")
    labelled = list(others)
    for c, s, b in safes:
        w, h = b[2] - b[0], b[3] - b[1]
        crop = img.crop((max(0, b[0] - .25 * w), max(0, b[1] - .25 * h), min(img.width, b[2] + .25 * w), min(img.height, b[3] + .25 * h)))
        is_open, _ = safe_is_open(crop.resize((crop.width * 2, crop.height * 2)))
        labelled.append(("safe_open" if is_open else "safe_closed", s, b))
    final[f] = {**rec, "labels": labelled}
    stats.update(c for c, _, _ in labelled)
    if (k + 1) % 200 == 0:
        print(f"{k + 1}/{len(raw)}  {(time.time() - t0) / 60:.1f} мин", dict(stats), flush=True)
print("рамок по классам:", dict(stats))

for split in ("train", "valid"):
    coco = {"images": [], "annotations": [], "categories": [{"id": i + 1, "name": c} for i, c in enumerate(CLASSES)]}
    for f, rec in final.items():
        if rec["split"] != split:
            continue
        img_id = len(coco["images"]) + 1
        coco["images"].append({"id": img_id, "file_name": Path(f).name, "width": rec["w"], "height": rec["h"]})
        for c, s, b in rec["labels"]:
            coco["annotations"].append({"id": len(coco["annotations"]) + 1, "image_id": img_id, "category_id": CLASSES.index(c) + 1,
                                        "bbox": [b[0], b[1], b[2] - b[0], b[3] - b[1]]})
    with zipfile.ZipFile(OUT / f"{split}_auto.zip", "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("annotations/instances_default.json", json.dumps(coco, ensure_ascii=False))
    print(f"{split}: {len(coco['images'])} кадров, {len(coco['annotations'])} рамок -> {OUT / f'{split}_auto.zip'}")
del qwen; torch.cuda.empty_cache()
'''),
    ("md", "## 3. Глазами: кадры с деньгами и открытым сейфом"),
    ("code", r'''
col = {"cash": "lime", "safe_open": "red", "safe_closed": "orange", "collection_bag": "green",
       "counting_machine": "cyan", "phone": "yellow", "handbag": "blue", "cup": "white"}
pick = [f for f, r in final.items() if any(c in ("cash", "safe_open") for c, _, _ in r["labels"])]
pick = pick[:: max(1, len(pick) // 12)][:12]
fig, axes = plt.subplots(2, 6, figsize=(24, 14))
for ax, f in zip(axes.ravel(), pick):
    im = Image.open(DS / "dataset" / f).convert("RGB"); d = ImageDraw.Draw(im)
    for c, s, b in final[f]["labels"]:
        d.rectangle(b, outline=col.get(c, "white"), width=3); d.text((b[0] + 2, b[1] + 2), c, fill=col.get(c, "white"))
    ax.imshow(im); ax.set_title(Path(f).name, fontsize=9); ax.axis("off")
plt.tight_layout(); plt.show()
'''),
    ("md", """
## Дальше
1. `03_runpod_detector.ipynb` — `MODE = "auto"` (уже стоит по умолчанию) → Run All (~15 мин).
2. `04_rules_eval.ipynb` — `MODE = "auto"` → Run All. Сравнить recall / false_alarms с черновиком (4/21, 0/19).
"""),
]

out = Path(sys.argv[1])
out.write_text(json.dumps(nb(CELLS), indent=1, ensure_ascii=False) + "\n")
print("written", out)

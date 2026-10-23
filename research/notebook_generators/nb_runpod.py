import json
import sys
from pathlib import Path


def nb(cells):
    out = []
    for i, (kind, src) in enumerate(cells):
        src = src.strip("\n")
        lines = [l + "\n" for l in src.split("\n")]
        lines[-1] = lines[-1].rstrip("\n")
        cell = {"cell_type": "markdown" if kind == "md" else "code", "id": f"cell-{i:02d}", "metadata": {}, "source": lines}
        if kind != "md":
            cell.update(execution_count=None, outputs=[])
        out.append(cell)
    return {"cells": out, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                                       "language_info": {"name": "python"}}, "nbformat": 4, "nbformat_minor": 5}


CELLS = [
    ("md", """
# 03 — RunPod: дообучение детектора + таймлайны по видео

1. Окружение и данные.
2. COCO (экспорт CVAT) → формат YOLO, проверка разметки.
3. Дообучение **YOLO11m** на наши классы с сильными аугментациями (финальное видео — под другим углом).
4. Оценка на valid (по классам) + прокси-тест на поворот/масштаб.
5. Таймлайны по всем 42 видео: люди + трекинг (**YOLO11m + BoT-SORT**), поза (**YOLO11m-pose**), наши объекты.
   Это вход для правил. Плюс замер времени на видео (бюджет ≤ 3 мин).

**Что загрузить в `/workspace/data/`** (JupyterLab → Upload, или `runpodctl send`):

| Файл | Куда |
|---|---|
| `datathon26_detector_dataset.zip` | `/workspace/data/` |
| папка `open/` (`videos/*.mov` + `open_labels.xlsx`), можно zip-ом | `/workspace/data/open/` |
| экспорт разметки: Label Studio JSON **или** CVAT COCO 1.0 | `/workspace/data/labels/train_ls.json`, `valid_ls.json` (или `data/cvat/train_coco.zip`, `valid_coco.zip`) |
| (для пробного прогона) `valid_coco1.0.zip`, `train_p1_coco1.0.zip` из `data/processed/prelabels` | `/workspace/data/prelabels/` |

`MODE = "draft"` обучает на автоматических черновых рамках — **только чтобы проверить, что всё работает**, метрики в этом режиме ничего не значат.
"""),
    ("md", "## 0. Параметры"),
    ("code", r'''
import os
from pathlib import Path

WORK   = Path(os.getenv("WORK", "/workspace"))
DATA   = WORK / "data"
MODE   = os.getenv("MODE", "auto")              # "auto" (авторазметка v2) | "cvat" (ручная разметка) | "draft" (черновик)
DEVICE = os.getenv("DEVICE", "0")               # "0" = первая GPU; "mps"/"cpu" для локальной проверки

DATASET_ZIP = DATA / "datathon26_detector_dataset.zip"
VIDEOS_DIR  = DATA / "open" / "videos"
LABELS_XLSX = DATA / "open" / "open_labels.xlsx"
def _first(*paths):
    return next((p for p in paths if p.exists()), paths[0])

COCO_SRC = {
    # настоящая разметка: экспорт Label Studio (JSON) или CVAT (COCO 1.0 zip) — берётся тот, что есть
    "cvat":  {"train": _first(DATA / "labels" / "train_ls.json", DATA / "cvat" / "train_coco.zip"),
              "valid": _first(DATA / "labels" / "valid_ls.json", DATA / "cvat" / "valid_coco.zip")},
    "draft": {"train": DATA / "prelabels" / "train_p1_coco1.0.zip", "valid": DATA / "prelabels" / "valid_coco1.0.zip"},
    # авторазметка v2: OWLv2-large (деньги) + Qwen (сейф открыт/закрыт) на всех 814 кадрах
    "auto":  {"train": DATA / "autolabels" / "train_auto.zip", "valid": DATA / "autolabels" / "valid_auto.zip"},
}[MODE]

CLASSES = ["cash", "collection_bag", "safe_open", "safe_closed", "counting_machine",
           "id_document", "phone", "handbag", "cup"]
INCLUDE_EMPTY = False   # кадры без единой рамки в обучение не берём: в CVAT «не размечен» и «пусто» выглядят одинаково

BASE_MODEL = os.getenv("BASE_MODEL", "yolo11m.pt")
EPOCHS = int(os.getenv("EPOCHS", 80))
IMGSZ  = int(os.getenv("IMGSZ", 896))           # кадры 464x848, мелкие объекты (купюры, телефон) -> крупный вход
BATCH  = int(os.getenv("BATCH", 16))
SEED   = 42

TIMELINE_FPS  = float(os.getenv("TIMELINE_FPS", 4))   # кадров в секунду для таймлайнов
LIMIT_VIDEOS  = int(os.getenv("LIMIT_VIDEOS", 0))     # 0 = все видео

RUNS   = WORK / "runs"
MODELS = WORK / "models"
YOLO_DS = WORK / "yolo_ds"
TIMELINES = WORK / "timelines"
for p in (RUNS, MODELS, YOLO_DS, TIMELINES):
    p.mkdir(parents=True, exist_ok=True)
os.chdir(WORK)   # скачанные веса и служебные папки Ultralytics -> в /workspace, а не рядом с ноутбуком
'''),
    ("md", "## 1. Окружение"),
    ("code", r'''
import subprocess, sys
# в образе RunPod PyTorch нет pandas, а свежий ultralytics его больше не тянет -> ставим явно
r = subprocess.run([sys.executable, "-m", "pip", "install", "-q", "ultralytics==8.4.180", "pandas", "matplotlib",
                    "supervision", "av", "openpyxl", "pyarrow", "lap"], capture_output=True, text=True)
print(r.stdout[-2000:], r.stderr[-2000:])
assert r.returncode == 0, "pip install не прошёл, см. вывод выше"
import torch
print("torch", torch.__version__, "| cuda", torch.cuda.is_available(),
      "|", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "no GPU")
if torch.cuda.is_available():
    free, total = torch.cuda.mem_get_info(); print(f"VRAM free {free/1e9:.1f} / {total/1e9:.1f} GB")
print(subprocess.run(["df", "-h", str(WORK)], capture_output=True, text=True).stdout)
'''),
    ("code", r'''
import json, re, shutil, time, unicodedata, zipfile
from collections import defaultdict
from urllib.parse import unquote

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
from IPython.display import display


def nfc(s):
    return unicodedata.normalize("NFC", str(s))


def norm_vid(name):
    """'Х-12' (кириллица) / 'X-12' / 'Х-12.mov' -> 'X-12'."""
    s = nfc(name).strip().replace("Х", "X").replace("х", "X").replace("x", "X")
    return re.sub(r"\.mov$", "", s, flags=re.I)


def unzip_utf8(src, dst):
    """Распаковка с кириллицей в именах (zip без UTF-8 флага даёт кракозябры в zipfile)."""
    with zipfile.ZipFile(src) as z:
        for info in z.infolist():
            name = info.filename
            if not info.flag_bits & 0x800:
                try:
                    name = name.encode("cp437").decode("utf-8")
                except (UnicodeEncodeError, UnicodeDecodeError):
                    pass
            target = Path(dst) / nfc(name)
            if info.is_dir():
                target.mkdir(parents=True, exist_ok=True)
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            with z.open(info) as fsrc, open(target, "wb") as fdst:
                shutil.copyfileobj(fsrc, fdst)
'''),
    ("md", "## 2. Данные"),
    ("code", r'''
DS = DATA / "detds"
if not (DS / "dataset" / "manifest.csv").exists():
    unzip_utf8(DATASET_ZIP, DS)
if not VIDEOS_DIR.exists() and (DATA / "open.zip").exists():
    unzip_utf8(DATA / "open.zip", DATA)

manifest = pd.read_csv(DS / "dataset" / "manifest.csv")
notes = pd.read_csv(DS / "video_notes.csv")
print("кадров:", len(manifest), "| по сплитам:", manifest.split.value_counts().to_dict())
videos = sorted(VIDEOS_DIR.glob("*.mov"))
print("видео:", len(videos))
missing = [p for p in COCO_SRC.values() if not Path(p).exists()]
assert not missing, f"нет файлов разметки: {missing}"
'''),
    ("md", "## 3. COCO → YOLO"),
    ("code", r'''
CLASS_ID = {c: i + 1 for i, c in enumerate(CLASSES)}


def ls_to_coco(tasks):
    """Label Studio JSON export -> COCO. Only tasks with a saved annotation (= reviewed) are kept."""
    coco = {"images": [], "annotations": [], "categories": [{"id": i, "name": c} for c, i in CLASS_ID.items()]}
    for task in tasks:
        done = [a for a in task.get("annotations", []) if not a.get("was_cancelled")]
        if not done:
            continue
        ann = max(done, key=lambda a: a.get("updated_at") or a.get("created_at") or "")
        d = task["data"]; img_id = len(coco["images"]) + 1
        coco["images"].append({"id": img_id, "file_name": d["fname"], "width": d["width"], "height": d["height"],
                               "reviewed_empty": not ann["result"]})
        for r in ann["result"]:
            if r.get("type") != "rectanglelabels":
                continue
            v, W, H = r["value"], r.get("original_width", d["width"]), r.get("original_height", d["height"])
            for lbl in v["rectanglelabels"]:
                if lbl in CLASS_ID:
                    coco["annotations"].append({"id": len(coco["annotations"]) + 1, "image_id": img_id,
                                                "category_id": CLASS_ID[lbl],
                                                "bbox": [v["x"] * W / 100, v["y"] * H / 100, v["width"] * W / 100, v["height"] * H / 100]})
    return coco


def load_coco(src):
    src = Path(src)
    if src.suffix == ".zip":
        with zipfile.ZipFile(src) as z:
            name = next(n for n in z.namelist() if n.endswith(".json"))
            data = json.loads(z.read(name))
    else:
        data = json.loads(src.read_text())
    return ls_to_coco(data) if isinstance(data, list) else data


SRC_INDEX = {nfc(p.name): p for p in (DS / "dataset" / "images").rglob("*.jpg")}


def coco_to_yolo(coco, split):
    img_dir, lbl_dir = YOLO_DS / "images" / split, YOLO_DS / "labels" / split
    for d in (img_dir, lbl_dir):
        shutil.rmtree(d, ignore_errors=True); d.mkdir(parents=True)
    cat = {c["id"]: c["name"] for c in coco["categories"]}
    unknown = sorted(set(cat.values()) - set(CLASSES))
    anns = defaultdict(list)
    for a in coco["annotations"]:
        anns[a["image_id"]].append(a)
    stats = defaultdict(int); not_found = []
    for im in coco["images"]:
        name = nfc(Path(unquote(im["file_name"])).name)
        src = SRC_INDEX.get(name)
        if src is None:
            not_found.append(name); continue
        W, H = im["width"], im["height"]
        lines = []
        for a in anns[im["id"]]:
            cname = cat[a["category_id"]]
            x, y, w, h = a["bbox"]
            if cname not in CLASSES or w <= 1 or h <= 1:
                continue
            x0, y0 = max(0, x), max(0, y); x1, y1 = min(W, x + w), min(H, y + h)
            lines.append(f"{CLASSES.index(cname)} {(x0+x1)/2/W:.6f} {(y0+y1)/2/H:.6f} {(x1-x0)/W:.6f} {(y1-y0)/H:.6f}")
            stats[cname] += 1
        if not lines and not (INCLUDE_EMPTY or im.get("reviewed_empty")):
            stats["_skipped_empty"] += 1; continue
        shutil.copy2(src, img_dir / name)
        (lbl_dir / (Path(name).stem + ".txt")).write_text("\n".join(lines))
        stats["_images"] += 1
    if unknown:
        print(f"[{split}] метки не из списка CLASSES (пропущены): {unknown}")
    if not_found:
        print(f"[{split}] нет картинок для {len(not_found)} записей, напр. {not_found[:3]}")
    return dict(stats)


stats = {split: coco_to_yolo(load_coco(src), split) for split, src in COCO_SRC.items()}
tbl = pd.DataFrame(stats).reindex(["_images", "_skipped_empty"] + CLASSES).fillna(0).astype(int)
display(tbl)
weak = [c for c in CLASSES if tbl.loc[c, "train"] < 15]
print("мало примеров в train (<15) — класс будет учиться плохо:", weak)

DATA_YAML = YOLO_DS / "data.yaml"
DATA_YAML.write_text(f"path: {YOLO_DS}\ntrain: images/train\nval: images/valid\nnames:\n" +
                     "".join(f"  {i}: {c}\n" for i, c in enumerate(CLASSES)))
print(DATA_YAML.read_text())
'''),
    ("code", r'''
# Глазами: 6 случайных кадров train с рамками из разметки
rng = np.random.default_rng(SEED)
imgs = sorted((YOLO_DS / "images" / "train").glob("*.jpg"))
fig, axes = plt.subplots(1, 6, figsize=(24, 8))
for ax, p in zip(axes, rng.choice(imgs, size=min(6, len(imgs)), replace=False)):
    im = Image.open(p).convert("RGB"); d = ImageDraw.Draw(im); W, H = im.size
    for line in (YOLO_DS / "labels" / "train" / (p.stem + ".txt")).read_text().splitlines():
        c, cx, cy, w, h = line.split(); cx, cy, w, h = map(float, (cx, cy, w, h))
        d.rectangle([(cx-w/2)*W, (cy-h/2)*H, (cx+w/2)*W, (cy+h/2)*H], outline="red", width=3)
        d.text(((cx-w/2)*W+3, (cy-h/2)*H+3), CLASSES[int(c)], fill="red")
    ax.imshow(im); ax.set_title(p.name, fontsize=8); ax.axis("off")
plt.tight_layout(); plt.show()
'''),
    ("md", """
## 4. Дообучение YOLO11m

Аугментации сильные специально: финальное видео будет **под другим углом**. Поворот, перспектива, масштаб, сдвиг, яркость.
"""),
    ("code", r'''
from ultralytics import YOLO

RUN_NAME = f"det_{MODE}"
t0 = time.time()
model = YOLO(BASE_MODEL)
model.train(
    data=str(DATA_YAML), epochs=EPOCHS, imgsz=IMGSZ, batch=BATCH, device=DEVICE, seed=SEED,
    project=str(RUNS), name=RUN_NAME, exist_ok=True, patience=25, cos_lr=True, cache="ram", workers=8, plots=True,
    # геометрия: другой ракурс камеры
    degrees=15, translate=0.15, scale=0.5, shear=5, perspective=0.0007, fliplr=0.5,
    # фотометрия: другое освещение
    hsv_h=0.015, hsv_s=0.6, hsv_v=0.5,
    mosaic=1.0, mixup=0.1, close_mosaic=10,
)
print(f"обучение заняло {(time.time()-t0)/60:.1f} мин")
BEST = RUNS / RUN_NAME / "weights" / "best.pt"
shutil.copy2(BEST, MODELS / f"cash_yolo11m_{MODE}.pt")
print("веса:", MODELS / f"cash_yolo11m_{MODE}.pt")
'''),
    ("md", "## 5. Оценка на valid"),
    ("code", r'''
det = YOLO(str(BEST))
m = det.val(data=str(DATA_YAML), imgsz=IMGSZ, device=DEVICE, split="val", plots=True, verbose=False,
            project=str(RUNS), name=f"{RUN_NAME}_val", exist_ok=True)
rows = []
for k, ci in enumerate(m.box.ap_class_index):
    p, r, ap50, ap = m.box.class_result(k)
    rows.append({"class": CLASSES[int(ci)], "precision": p, "recall": r, "mAP50": ap50, "mAP50-95": ap})
per_class = pd.DataFrame(rows).set_index("class").round(3)
display(per_class)
print(f"ИТОГО mAP50={m.box.map50:.3f}  mAP50-95={m.box.map:.3f}")
per_class.to_csv(RUNS / RUN_NAME / "valid_per_class.csv")
'''),
    ("code", r'''
# Предсказания на 6 кадрах valid (глазами: что модель реально видит)
vimgs = sorted((YOLO_DS / "images" / "valid").glob("*.jpg"))
pick = rng.choice(vimgs, size=min(6, len(vimgs)), replace=False)
res = det.predict([str(p) for p in pick], imgsz=IMGSZ, conf=0.25, device=DEVICE, verbose=False)
fig, axes = plt.subplots(1, len(res), figsize=(24, 8))
for ax, r in zip(np.atleast_1d(axes), res):
    ax.imshow(r.plot()[:, :, ::-1]); ax.set_title(Path(r.path).name, fontsize=8); ax.axis("off")
plt.tight_layout(); plt.show()
'''),
    ("md", """
### Прокси-тест на другой ракурс

Копия valid с поворотом ±12°, масштабом 0.8–1.1 и сдвигом; рамки пересчитаны. Падение mAP показывает, насколько детектор держится при смене угла.
"""),
    ("code", r'''
import cv2

def make_shifted_valid(angle_max=12, seed=0):
    rng_ = np.random.default_rng(seed)
    root = WORK / "yolo_ds_shift"
    for sub in ("images/val", "labels/val"):
        shutil.rmtree(root / sub, ignore_errors=True); (root / sub).mkdir(parents=True)
    for p in sorted((YOLO_DS / "images" / "valid").glob("*.jpg")):
        img = cv2.imread(str(p)); H, W = img.shape[:2]
        a, s = rng_.uniform(-angle_max, angle_max), rng_.uniform(0.8, 1.1)
        M = cv2.getRotationMatrix2D((W / 2, H / 2), a, s)
        M[:, 2] += rng_.uniform(-0.05, 0.05, 2) * (W, H)
        out = cv2.warpAffine(img, M, (W, H), borderMode=cv2.BORDER_REFLECT)
        lines = []
        for line in (YOLO_DS / "labels" / "valid" / (p.stem + ".txt")).read_text().splitlines():
            c, cx, cy, w, h = line.split(); cx, cy, w, h = float(cx) * W, float(cy) * H, float(w) * W, float(h) * H
            pts = np.array([[cx-w/2, cy-h/2, 1], [cx+w/2, cy-h/2, 1], [cx+w/2, cy+h/2, 1], [cx-w/2, cy+h/2, 1]]) @ M.T
            x0, y0 = pts.min(0).clip(0, [W, H]); x1, y1 = pts.max(0).clip(0, [W, H])
            if (x1 - x0) > 3 and (y1 - y0) > 3:
                lines.append(f"{c} {(x0+x1)/2/W:.6f} {(y0+y1)/2/H:.6f} {(x1-x0)/W:.6f} {(y1-y0)/H:.6f}")
        cv2.imwrite(str(root / "images/val" / p.name), out)
        (root / "labels/val" / (p.stem + ".txt")).write_text("\n".join(lines))
    y = root / "data.yaml"
    y.write_text(f"path: {root}\ntrain: images/val\nval: images/val\nnames:\n" + "".join(f"  {i}: {c}\n" for i, c in enumerate(CLASSES)))
    return y

m_shift = det.val(data=str(make_shifted_valid()), imgsz=IMGSZ, device=DEVICE, plots=False, verbose=False,
                  project=str(RUNS), name=f"{RUN_NAME}_val_shift", exist_ok=True)
print(f"mAP50: исходный ракурс {m.box.map50:.3f} -> сдвинутый {m_shift.box.map50:.3f}  (падение {m.box.map50 - m_shift.box.map50:+.3f})")
'''),
    ("md", """
## 6. Таймлайны по видео

Для каждого видео с частотой `TIMELINE_FPS`:
- **люди** — `yolo11m.pt` (COCO) + трекер BoT-SORT: id трека и рамка;
- **поза** — `yolo11m-pose.pt`: запястья, нос, бёдра (для «руки к карману/лицу», «пересчёт»);
- **наши объекты** — дообученный детектор.

Сохраняется в `/workspace/timelines/<видео>_{persons,pose,objects}.parquet` — это вход для правил.
"""),
    ("code", r'''
def load_labels():
    df = pd.read_excel(LABELS_XLSX)
    df.columns = ["file", "ts", "label"]
    def parse_ts(v):
        if pd.isna(v):
            return None
        s = str(v).lower(); mm = re.search(r"(\d+)\s*мин", s); ss = re.search(r"(\d+)\s*сек", s)
        return float((int(mm.group(1)) * 60 if mm else 0) + (int(ss.group(1)) if ss else 0))
    df["video_id"] = df.file.map(norm_vid)
    df["violation"] = df.label.str.strip().str.lower().eq("нарушение")
    df["t_sec"] = df.ts.map(parse_ts)
    files = {norm_vid(p.name): p for p in VIDEOS_DIR.glob("*.mov")}
    df["path"] = df.video_id.map(files)
    return df[["video_id", "violation", "t_sec", "path"]]

labels = load_labels()
assert labels.path.notna().all(), labels[labels.path.isna()]
if LIMIT_VIDEOS:
    labels = labels.head(LIMIT_VIDEOS)

t_load = time.time()
person_model = YOLO("yolo11m.pt")
pose_model = YOLO("yolo11m-pose.pt")
obj_model = YOLO(str(MODELS / f"cash_yolo11m_{MODE}.pt"))
LOAD_S = time.time() - t_load
print(f"модели загружены за {LOAD_S:.1f} c")
KP = {"nose": 0, "l_wrist": 9, "r_wrist": 10, "l_hip": 11, "r_hip": 12}
'''),
    ("code", r'''
def video_timeline(path):
    cap = cv2.VideoCapture(str(path)); fps = cap.get(cv2.CAP_PROP_FPS) or 30.0; cap.release()
    step = max(1, round(fps / TIMELINE_FPS))
    common = dict(stream=True, vid_stride=step, device=DEVICE, verbose=False, imgsz=IMGSZ)
    persons, pose, objects = [], [], []
    # люди + трекинг (трекер сбрасывается на каждом новом источнике)
    for k, r in enumerate(person_model.track(source=str(path), classes=[0], conf=0.3, tracker="botsort.yaml", **common)):
        t = k * step / fps
        ids = r.boxes.id.int().cpu().tolist() if r.boxes.id is not None else [-1] * len(r.boxes)
        for tid, b, c in zip(ids, r.boxes.xyxy.cpu().numpy(), r.boxes.conf.cpu().numpy()):
            persons.append({"t": t, "track": tid, "x0": b[0], "y0": b[1], "x1": b[2], "y1": b[3], "conf": float(c)})
    for k, r in enumerate(pose_model.predict(source=str(path), conf=0.3, **common)):
        t = k * step / fps
        if r.keypoints is None:
            continue
        for b, kp in zip(r.boxes.xyxy.cpu().numpy(), r.keypoints.data.cpu().numpy()):
            row = {"t": t, "x0": b[0], "y0": b[1], "x1": b[2], "y1": b[3]}
            for name, i in KP.items():
                row[f"{name}_x"], row[f"{name}_y"], row[f"{name}_c"] = kp[i]
            pose.append(row)
    for k, r in enumerate(obj_model.predict(source=str(path), conf=0.25, **common)):
        t = k * step / fps
        for b, c, s in zip(r.boxes.xyxy.cpu().numpy(), r.boxes.cls.cpu().numpy(), r.boxes.conf.cpu().numpy()):
            objects.append({"t": t, "cls": CLASSES[int(c)], "x0": b[0], "y0": b[1], "x1": b[2], "y1": b[3], "conf": float(s)})
    return pd.DataFrame(persons), pd.DataFrame(pose), pd.DataFrame(objects)


timing = []
for r in labels.itertuples():
    t0 = time.time()
    P, K, O = video_timeline(r.path)
    dt = time.time() - t0
    for name, df_ in (("persons", P), ("pose", K), ("objects", O)):
        df_.to_parquet(TIMELINES / f"{r.video_id}_{name}.parquet", index=False)
    timing.append({"video_id": r.video_id, "sec": round(dt, 1), "persons": len(P), "objects": len(O)})
    print(timing[-1], flush=True)
timing = pd.DataFrame(timing)
'''),
    ("code", r'''
def video_seconds(path):
    cap = cv2.VideoCapture(str(path)); n, fps = cap.get(cv2.CAP_PROP_FRAME_COUNT), cap.get(cv2.CAP_PROP_FPS) or 30.0
    cap.release(); return n / fps

dur = {r.video_id: video_seconds(r.path) for r in labels.itertuples()}
timing["video_s"] = timing.video_id.map(dur).round(1)
print(f"время на видео: p50 {timing.sec.median():.1f} c, худшее {timing.sec.max():.1f} c "
      f"(видео {timing.loc[timing.sec.idxmax(), 'video_id']}, {timing.loc[timing.sec.idxmax(), 'video_s']} c)")
print(f"худший случай с загрузкой моделей: {timing.sec.max() + LOAD_S:.1f} c из бюджета 180 c")
timing.to_csv(TIMELINES / "timing.csv", index=False)
'''),
    ("md", "### Таймлайны глазами: люди и объекты во времени, красная линия — таймкод нарушения"),
    ("code", r'''
def plot_timeline(vid):
    row = labels[labels.video_id == vid].iloc[0]
    P = pd.read_parquet(TIMELINES / f"{vid}_persons.parquet")
    O = pd.read_parquet(TIMELINES / f"{vid}_objects.parquet")
    fig, ax = plt.subplots(2, 1, figsize=(14, 4.5), sharex=True)
    if len(P):
        P.groupby("t").size().plot(ax=ax[0], drawstyle="steps-post"); ax[0].set_ylabel("людей")
    if len(O):
        for i, (c, g) in enumerate(O.groupby("cls")):
            ax[1].scatter(g.t, [i] * len(g), s=6, label=c)
        ax[1].set_yticks(range(O.cls.nunique())); ax[1].set_yticklabels(sorted(O.cls.unique()))
    for a in ax:
        if row.violation:
            a.axvline(row.t_sec, color="red", lw=2)
    ax[0].set_title(f"{vid}  {'НАРУШЕНИЕ @ %.0f c' % row.t_sec if row.violation else 'без нарушения'}")
    ax[1].set_xlabel("секунды"); plt.tight_layout(); plt.show()

for vid in labels.video_id.head(6):
    plot_timeline(vid)
'''),
    ("md", """
## 7. Что забрать с пода

- `/workspace/models/cash_yolo11m_<MODE>.pt` — дообученный детектор.
- `/workspace/runs/det_<MODE>/` — графики обучения, `valid_per_class.csv`, матрица ошибок.
- `/workspace/timelines/` — таймлайны для правил + `timing.csv`.

**Проверка офлайн:** все веса — локальные `.pt`, интернет при инференсе не нужен (`HF_HUB_OFFLINE=1` не мешает).
"""),
]

if __name__ == '__main__':
    out = Path(sys.argv[1])
    out.write_text(json.dumps(nb(CELLS), indent=1, ensure_ascii=False) + "\n")
    print("written", out)

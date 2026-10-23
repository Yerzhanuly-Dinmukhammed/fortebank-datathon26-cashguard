import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from nb_runpod import nb  # noqa: E402

CELLS = [
    ("md", """
# 04 — Правила, оценка по видео и `analyze()`

Берёт таймлайны из `03` (`/workspace/timelines`), применяет правила (`src/rules.py`) и считает:
- **recall** по видео (найдено нарушений) и **ложные срабатывания** на чистых видео;
- **ошибку таймкода** и совпадение типа с `video_notes.csv`.

Дубликаты видео (побайтово одинаковые файлы: Х-4 = Х-47, Х-5 = Х-48) объединяются в одно видео со всеми таймкодами.
В конце — `analyze()` (видео → JSON по ТЗ) на нескольких роликах с замером времени.

**Загрузить:** `src.zip` → `/workspace/` (распакуется сам).
"""),
    ("md", "## 0. Параметры"),
    ("code", r'''
import os, sys, zipfile
from pathlib import Path

WORK  = Path(os.getenv("WORK", "/workspace"))
MODE  = os.getenv("MODE", "auto")                 # какой детектор: "draft" | "cvat" (как в 03)
DEVICE = os.getenv("DEVICE", "0")
THRESHOLD = float(os.getenv("THRESHOLD", 0.5))     # порог правил (0.5: 9/21 найдено, 2/19 ложных на авторазметке)

DATA = WORK / "data"
TIMELINES = WORK / "timelines"
VIDEOS_DIR = DATA / "open" / "videos"
LABELS_XLSX = DATA / "open" / "open_labels.xlsx"
NOTES_CSV = DATA / "detds" / "video_notes.csv"
OBJ_WEIGHTS = WORK / "models" / f"cash_yolo11m_{MODE}.pt"
APP = WORK / "app"
RESULTS = WORK / "results"; RESULTS.mkdir(parents=True, exist_ok=True)

if (WORK / "src.zip").exists():                     # всегда берём свежий код из src.zip
    zipfile.ZipFile(WORK / "src.zip").extractall(APP)
sys.path.insert(0, str(APP))
os.chdir(WORK)
print("код:", sorted(p.name for p in (APP / "src").iterdir()))
'''),
    ("md", "## 1. Разметка + объединение дубликатов"),
    ("code", r'''
import re, unicodedata, json
import numpy as np
import pandas as pd
from IPython.display import display
from src.evaluate import merge_duplicates, run_rules, score

def norm_vid(s):
    s = unicodedata.normalize("NFC", str(s)).strip().replace("Х", "X").replace("х", "X").replace("x", "X")
    return re.sub(r"\.mov$", "", s, flags=re.I)

def parse_ts(v):
    if pd.isna(v):
        return None
    s = str(v).lower(); m = re.search(r"(\d+)\s*мин", s); c = re.search(r"(\d+)\s*сек", s)
    return float((int(m.group(1)) * 60 if m else 0) + (int(c.group(1)) if c else 0))

lab = pd.read_excel(LABELS_XLSX); lab.columns = ["file", "ts", "label"]
lab["video_id"] = lab.file.map(norm_vid)
lab["violation"] = lab.label.str.strip().str.lower().eq("нарушение")
lab["t_sec"] = lab.ts.map(parse_ts)
files = {norm_vid(p.name): p for p in VIDEOS_DIR.glob("*.mov")}
lab["path"] = lab.video_id.map(files)
notes = pd.read_csv(NOTES_CSV); notes["video_id"] = notes["видео"].map(norm_vid)
lab = lab.merge(notes[["video_id", "вероятный_тип"]].rename(columns={"вероятный_тип": "type"}), on="video_id", how="left")

videos = merge_duplicates(lab)
print(f"уникальных видео: {len(videos)} (нарушений {int(videos.violation.sum())}, чистых {int((~videos.violation).sum())})")
display(videos[videos.video_id.str.contains(r"\+")][["video_id", "violation", "t_list", "types"]])
'''),
    ("md", "## 2. Правила на всех видео: подбор порога"),
    ("code", r'''
events = run_rules(videos, TIMELINES)
sweep = pd.DataFrame([score(videos, events, thr)[0] for thr in [0.3, 0.35, 0.4, 0.5, 0.6, 0.7, 0.8]])
display(sweep)
'''),
    ("code", r'''
summary, per_video = score(videos, events, THRESHOLD)
print(summary)
pd.set_option("display.max_colwidth", 120)
display(per_video)
per_video.to_csv(RESULTS / f"rules_eval_{MODE}.csv", index=False)
(APP / "config.json").write_text(json.dumps({"threshold": THRESHOLD, "mode": MODE}))
'''),
    ("md", """
### Как читать
- **recall** — главное (пропуск нарушения для банка недопустим). **false_alarms** — сколько чистых видео помечено нарушением.
- Порог выбираем так, чтобы recall был максимальным при приемлемых false_alarms, и записываем в `THRESHOLD`.
- Пропущенные видео смотрим в таблице: какой сигнал не сработал (обычно — детектор не увидел `cash` / `safe_open`).
"""),
    ("md", "## 3. `analyze()`: видео → JSON по ТЗ, с замером времени"),
    ("code", r'''
import time, base64, io
import matplotlib.pyplot as plt
from PIL import Image
from src.pipeline import Analyzer

t0 = time.time()
analyzer = Analyzer(person_weights="yolo11m.pt", object_weights=OBJ_WEIGHTS, device=DEVICE, threshold=THRESHOLD)
print(f"модели загружены за {time.time()-t0:.1f} c; детектор объектов: {OBJ_WEIGHTS.name if analyzer.obj_model else 'НЕТ'}")

demo = ["X-46", "X-15", "X-3", "X-14"]       # 2 нарушения, чистое, самое длинное (142 c)
for vid in demo:
    res = analyzer.analyze(files[vid], job_id=vid)
    (RESULTS / f"{vid}.json").write_text(json.dumps(res, ensure_ascii=False))
    print(f"{vid}: {res['verdict']}, нарушений {len(res['violations'])}, {res['processing_sec']} c")
    for v in res["violations"]:
        print(f"   {v['timecode_sec']:6.1f} c  {v['type']}  ({v['confidence']})  {v['description']}")
        fig, axes = plt.subplots(1, len(v["evidence_frames"]), figsize=(10, 6))
        for ax, f in zip(np.atleast_1d(axes), v["evidence_frames"]):
            ax.imshow(Image.open(io.BytesIO(base64.b64decode(f["image_base64"])))); ax.set_title(f"{f['timestamp_sec']} c"); ax.axis("off")
        plt.suptitle(f"{vid}: {v['type']}"); plt.tight_layout(); plt.show()
'''),
    ("md", """
## Дальше
- После переобучения детектора на разметке CVAT: `MODE = "cvat"` в `03` и здесь → перезапуск. Цифры recall / false_alarms сравнить с текущими.
- `src/rules_norms.json` — заполняет ответственный за нормативку (документ, пункт, краткая трактовка).
- `Analyzer.analyze()` — то, что вызывает API сервиса.
"""),
]

out = Path(sys.argv[1])
out.write_text(json.dumps(nb(CELLS), indent=1, ensure_ascii=False) + "\n")
print("written", out)

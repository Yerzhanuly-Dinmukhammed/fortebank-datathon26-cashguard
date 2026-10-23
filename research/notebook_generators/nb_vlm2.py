import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from nb_runpod import nb  # noqa: E402

CELLS = [
    ("md", """
# 07 — Qwen3.5-9B: состояние после ухода кассира + перепроверка срабатываний правил

- **leftover** — кадры сразу после того, как кассир вышел: на столе что-то осталось / сейф открыт? (ловим «ушла, не убрав ценности»)
- **verify_cash / verify_handbag / verify_cup** — подтверждение срабатываний правил R1/R4 (убираем ложные)

Результат: `/workspace/results/vlm2_answers_Qwen3.5-9B.csv` → скачать и отдать ML.
"""),
    ("code", r'''
import os, sys, json, time, zipfile, re, unicodedata
from pathlib import Path
WORK = Path(os.getenv("WORK", "/workspace"))
MODEL_ID = os.getenv("MODEL_ID", "Qwen/Qwen3.5-9B")
os.environ.setdefault("HF_HOME", str(WORK / "hf_cache"))
APP, RESULTS = WORK / "app", WORK / "results"; RESULTS.mkdir(exist_ok=True)
zipfile.ZipFile(WORK / "src.zip").extractall(APP); sys.path.insert(0, str(APP)); os.chdir(WORK)
TIMELINES, VIDEOS_DIR = WORK / "timelines", WORK / "data" / "open" / "videos"

import pandas as pd, cv2
from PIL import Image
from src.rules import detect_events
import src.vlm_checks as V

def norm_vid(s):
    s = unicodedata.normalize("NFC", str(s)).strip().replace("Х", "X").replace("х", "X").replace("x", "X")
    return re.sub(r"\.mov$", "", s, flags=re.I)
files = {norm_vid(p.name): p for p in VIDEOS_DIR.glob("*.mov")}
keys = sorted({p.name.rsplit("_", 1)[0] for p in TIMELINES.glob("*_persons.parquet")})
cands = {}
for k in keys:
    P, O = pd.read_parquet(TIMELINES / f"{k}_persons.parquet"), pd.read_parquet(TIMELINES / f"{k}_objects.parquet")
    cands[k] = V.candidates2(P, O, 848, detect_events(P, O, 848))
print("видео:", len(keys), "| вопросов:", {q: sum(c["q"] == q for cs in cands.values() for c in cs) for q in V.QUESTIONS2})
'''),
    ("code", r'''
qwen = V.QwenYesNo(MODEL_ID)

def grab(path, t, side=640):
    cap = cv2.VideoCapture(str(path)); cap.set(cv2.CAP_PROP_POS_MSEC, max(t, 0) * 1000); ok, fr = cap.read(); cap.release()
    if not ok: return None
    im = Image.fromarray(cv2.cvtColor(fr, cv2.COLOR_BGR2RGB)); im.thumbnail((side, side)); return im

# проверка: X-24 ушла, оставив предметы (ждём yes); X-25 убрала всё и ушла (ждём no)
for vid, t, exp in [("X-24", 12.5, "yes"), ("X-25", 30.1, "no"), ("X-28", 13.6, "yes"), ("X-29", 13.6, "no")]:
    imgs = [im for im in (grab(files[vid], t + d) for d in (1, 2, 3)) if im is not None]
    print(f"{vid} leftover ожидаем {exp}: p_yes={qwen.p_yes(imgs, V.QUESTIONS2['leftover']):.2f}" if imgs else f"{vid}: нет кадров")
'''),
    ("code", r'''
rows, t0 = [], time.time()
for k, cs in cands.items():
    for c in cs:
        imgs = [im for im in (grab(files[k], x) for x in c["frames"]) if im is not None]
        if imgs:
            rows.append({"video": k, "q": c["q"], "t": c["t"], "rule": c.get("rule", ""),
                         "p_yes": qwen.p_yes(imgs, V.QUESTIONS2[c["q"]])})
out = RESULTS / f"vlm2_answers_{MODEL_ID.split('/')[-1]}.csv"
pd.DataFrame(rows).to_csv(out, index=False)
print(f"готово: {len(rows)} ответов за {(time.time() - t0) / 60:.1f} мин -> {out}")
'''),
]

out = Path(sys.argv[1])
out.write_text(json.dumps(nb(CELLS), indent=1, ensure_ascii=False) + "\n")
print("written", out)

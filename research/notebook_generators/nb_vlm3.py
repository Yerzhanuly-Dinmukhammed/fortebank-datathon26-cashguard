import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from nb_runpod import nb  # noqa: E402

CELLS = [
    ("md", """
# 08 — Qwen3.5-9B: плотный крупный ролик у сейфа перед уходом кассира

Для каждого ухода кассира: 12 кадров за 10 с до ухода, **вырезанная и увеличенная область сейфа** (вместе с кассиром рядом).
Вопрос: кладёт ли кассир что-то в сейф / закрывает ли дверцу. Ловим «ушла, не убрав ценности» (X-11, X-22, X-24, X-28).
Результат: `/workspace/results/vlm3_answers_Qwen3.5-9B.csv` → скачать и отдать ML.
"""),
    ("code", r'''
import os, sys, zipfile, re, unicodedata, time
from pathlib import Path
import numpy as np
WORK = Path(os.getenv("WORK", "/workspace")); MODEL_ID = os.getenv("MODEL_ID", "Qwen/Qwen3.5-9B")
os.environ.setdefault("HF_HOME", str(WORK / "hf_cache"))
APP, RESULTS = WORK / "app", WORK / "results"; RESULTS.mkdir(exist_ok=True)
zipfile.ZipFile(WORK / "src.zip").extractall(APP); sys.path.insert(0, str(APP)); os.chdir(WORK)
TIMELINES, VIDEOS_DIR = WORK / "timelines", WORK / "data" / "open" / "videos"
import pandas as pd, cv2
from PIL import Image
from src.rules import in_room, people_count, changes
import src.vlm_checks as V

def norm_vid(s):
    s = unicodedata.normalize("NFC", str(s)).strip().replace("Х", "X").replace("х", "X").replace("x", "X")
    return re.sub(r"\.mov$", "", s, flags=re.I)
files = {norm_vid(p.name): p for p in VIDEOS_DIR.glob("*.mov")}
QUESTION = ("These frames are a close-up of the area around the grey metal safe in a bank cash room, during the last "
            "10 seconds before the cashier leaves the room (frame 1 is earliest). Does the cashier put money or items "
            "into the safe, or close and lock the safe door, before leaving?")

cands = []
for p in sorted(TIMELINES.glob("*_persons.parquet")):
    k = p.name.rsplit("_", 1)[0]
    P, O = in_room(pd.read_parquet(p), 848), pd.read_parquet(TIMELINES / f"{k}_objects.parquet")
    S = O[O.cls.isin(["safe_closed", "safe_open"]) & (O.conf >= 0.5)]
    if not len(S): continue
    x0, y0, x1, y1 = S[["x0", "y0", "x1", "y1"]].median().values
    w, h = x1 - x0, y1 - y0
    box = (max(0, x0 - 1.2 * w), max(0, y0 - 1.0 * h), x1 + 1.2 * w, y1 + 1.0 * h)   # сейф + кассир рядом
    ts = np.sort(np.unique(np.r_[O.t.unique(), P.t.unique()])); n = people_count(P, ts)
    for t, a, b in changes(n):
        if a >= 1 and b == 0:
            cands.append({"video": k, "t": float(t), "box": box, "frames": list(np.linspace(max(t - 10, 0), t, 12))})
print("уходов:", len(cands))
'''),
    ("code", r'''
qwen = V.QwenYesNo(MODEL_ID)

def crop(path, t, box, side=448):
    cap = cv2.VideoCapture(str(path)); cap.set(cv2.CAP_PROP_POS_MSEC, max(t, 0) * 1000); ok, fr = cap.read(); cap.release()
    if not ok: return None
    im = Image.fromarray(cv2.cvtColor(fr, cv2.COLOR_BGR2RGB))
    im = im.crop(tuple(int(v) for v in (box[0], box[1], min(im.width, box[2]), min(im.height, box[3]))))
    s = side / max(im.size); return im.resize((max(1, int(im.width * s)), max(1, int(im.height * s))))

rows, t0 = [], time.time()
for c in cands:
    imgs = [im for im in (crop(files[c["video"]], x, c["box"]) for x in c["frames"]) if im is not None]
    if imgs:
        rows.append({"video": c["video"], "t": c["t"], "n_frames": len(imgs), "p_yes": qwen.p_yes(imgs, QUESTION)})
        print(rows[-1], flush=True)
out = RESULTS / f"vlm3_answers_{MODEL_ID.split('/')[-1]}.csv"
pd.DataFrame(rows).to_csv(out, index=False)
print(f"готово: {len(rows)} ответов за {(time.time() - t0) / 60:.1f} мин -> {out}")
'''),
]

out = Path(sys.argv[1])
out.write_text(json.dumps(nb(CELLS), indent=1, ensure_ascii=False) + "\n")
print("written", out)

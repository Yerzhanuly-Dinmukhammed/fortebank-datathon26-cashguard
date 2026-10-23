import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from nb_runpod import nb  # noqa: E402

CELLS = [
    ("md", """
# 09 — Стресс-тест: «другой ракурс»

Каждое видео искажается как будто снято другой камерой (поворот, зум, перспектива, другая яркость) и прогоняется
через **весь** `analyze()` (детекторы + правила + Qwen-9B), как это сделает жюри. Сравниваем с исходными 32/40.

~25–35 мин. Результат: `/workspace/results/stress_<aug>.csv` → скачать и отдать ML.
"""),
    ("code", r'''
import os, sys, zipfile, re, unicodedata, time, json
from pathlib import Path
import numpy as np, pandas as pd, cv2
WORK = Path(os.getenv("WORK", "/workspace")); os.environ.setdefault("HF_HOME", str(WORK / "hf_cache"))
APP, RESULTS, STRESS = WORK / "app", WORK / "results", WORK / "stress"
for d in (RESULTS, STRESS): d.mkdir(exist_ok=True)
zipfile.ZipFile(WORK / "src.zip").extractall(APP); sys.path.insert(0, str(APP)); os.chdir(WORK)
from src.evaluate import merge_duplicates
from src.pipeline import Analyzer

AUG = os.getenv("AUG", "combo")   # combo = поворот 8° + зум 0.85 + перспектива + темнее на 20%
VIDEOS_DIR = WORK / "data" / "open" / "videos"
def norm_vid(s):
    s = unicodedata.normalize("NFC", str(s)).strip().replace("Х", "X").replace("х", "X").replace("x", "X")
    return re.sub(r"\.mov$", "", s, flags=re.I)
def parse_ts(v):
    if pd.isna(v): return None
    s = str(v).lower(); m = re.search(r"(\d+)\s*мин", s); c = re.search(r"(\d+)\s*сек", s)
    return float((int(m.group(1)) * 60 if m else 0) + (int(c.group(1)) if c else 0))
lab = pd.read_excel(WORK / "data" / "open" / "open_labels.xlsx"); lab.columns = ["file", "ts", "label"]
lab["video_id"] = lab.file.map(norm_vid); lab["violation"] = lab.label.str.strip().str.lower().eq("нарушение"); lab["t_sec"] = lab.ts.map(parse_ts)
files = {norm_vid(p.name): p for p in VIDEOS_DIR.glob("*.mov")}; lab["path"] = lab.video_id.map(files)
videos = merge_duplicates(lab)
print("уникальных видео:", len(videos))
'''),
    ("code", r'''
def distort(src, dst, angle=8, zoom=0.85, persp=0.06, gain=0.8, seed=0):
    """Same geometric/photometric change for all frames of a video (a 'different camera')."""
    cap = cv2.VideoCapture(str(src)); fps = cap.get(cv2.CAP_PROP_FPS) or 30
    W, H = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    rng = np.random.default_rng(seed)
    M = cv2.getRotationMatrix2D((W / 2, H / 2), angle * rng.choice([-1, 1]), 1 / zoom)
    d = persp * np.array([W, H])
    src_pts = np.float32([[0, 0], [W, 0], [W, H], [0, H]])
    dst_pts = src_pts + np.float32(rng.uniform(-1, 1, (4, 2)) * d)
    Pm = cv2.getPerspectiveTransform(src_pts, dst_pts)
    out = cv2.VideoWriter(str(dst), cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H))
    while True:
        ok, fr = cap.read()
        if not ok: break
        fr = cv2.warpAffine(fr, M, (W, H), borderMode=cv2.BORDER_REFLECT)
        fr = cv2.warpPerspective(fr, Pm, (W, H), borderMode=cv2.BORDER_REFLECT)
        out.write(np.clip(fr.astype(np.float32) * gain, 0, 255).astype(np.uint8))
    cap.release(); out.release()

t0 = time.time(); (STRESS / AUG).mkdir(exist_ok=True)
for i, r in enumerate(videos.itertuples()):
    dst = STRESS / AUG / f"{r.key}.mp4"
    if not dst.exists():
        distort(files[r.key], dst, seed=i)
print(f"искажённые видео готовы за {(time.time() - t0) / 60:.1f} мин")
'''),
    ("code", r'''
an = Analyzer(person_weights=str(WORK / "yolo11m.pt"), pose_weights=str(WORK / "yolo11m-pose.pt"),
              object_weights=str(WORK / "models" / "cash_yolo11m_auto.pt"), device="0", vlm_model_id="Qwen/Qwen3.5-9B")
rows = []
for r in videos.itertuples():
    t1 = time.time()
    try:
        res = an.analyze(STRESS / AUG / f"{r.key}.mp4", job_id=r.key)
        ts = [v["timecode_sec"] for v in res["violations"]]
        err = min((abs(a - b) for a in ts for b in r.t_list), default=None) if r.violation and ts else None
        rows.append({"video": r.video_id, "true": r.violation, "pred": res["verdict"] == "violation", "t_err": err,
                     "types": "; ".join(v["type"] for v in res["violations"]), "sec": round(time.time() - t1, 1), "notes": ",".join(res.get("notes", []))})
    except Exception as e:
        rows.append({"video": r.video_id, "true": r.violation, "pred": None, "t_err": None, "types": f"ERROR {e}", "sec": round(time.time() - t1, 1), "notes": ""})
    print(rows[-1], flush=True)
df = pd.DataFrame(rows); df.to_csv(RESULTS / f"stress_{AUG}.csv", index=False)
pos, neg = df[df.true], df[~df.true]
tp, fp = int((pos.pred == True).sum()), int((neg.pred == True).sum())
print(f"\n{AUG}: верно {tp + len(neg) - fp}/{len(df)} | найдено {tp}/{len(pos)} | ложных {fp}/{len(neg)} | "
      f"таймкод медиана {pos.t_err.median():.1f} c | время: медиана {df.sec.median():.0f} c, макс {df.sec.max():.0f} c | ошибок {df.pred.isna().sum()}")
'''),
]

out = Path(sys.argv[1])
out.write_text(json.dumps(nb(CELLS), indent=1, ensure_ascii=False) + "\n")
print("written", out)

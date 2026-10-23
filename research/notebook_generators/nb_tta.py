import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from nb_runpod import nb  # noqa: E402

CELLS = [
    ("md", """
# 11 — TTA (отражение + несколько масштабов) для детекторов людей и предметов

Один эксперимент на устойчивость к новому ракурсу. Тот же `analyze()`, но `Analyzer(tta=True)`.
Прогон на **чистых** 40 видео и на **искажённых** (из 09). Сравнение с базой: чистые 32/40, стресс 28/40.

**Перед запуском:** сервис на порту 8000 и другие ноутбуки с Qwen должны быть остановлены (GPU 22 ГБ на одну Qwen-9B):
`pkill -f "uvicorn serve.app:app"` и Kernel → Shut Down All Kernels. ~15–25 мин.

**Правило решения:** включаем TTA в сервис, только если стресс ≥ 30/40 **и** чистые ≥ 32/40 **и** ложных не больше, чем было.
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

AUG = "combo"   # искажённые видео уже лежат в /workspace/stress/combo (сделаны в 09)
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
assert (STRESS / AUG).exists() and len(list((STRESS / AUG).glob("*.mp4"))) == len(videos), "сначала прогоните 09 (нужны искажённые видео)"
an = Analyzer(person_weights=str(WORK / "yolo11m.pt"), pose_weights=str(WORK / "yolo11m-pose.pt"),
              object_weights=str(WORK / "models" / "cash_yolo11m_auto.pt"), device="0", vlm_model_id="Qwen/Qwen3.5-9B",
              tta=True)

def run(name, path_of):
    rows = []
    for r in videos.itertuples():
        t1 = time.time()
        try:
            res = an.analyze(path_of(r), job_id=r.key)
            ts = [v["timecode_sec"] for v in res["violations"]]
            err = min((abs(a - b) for a in ts for b in r.t_list), default=None) if r.violation and ts else None
            rows.append({"video": r.video_id, "true": r.violation, "pred": res["verdict"] == "violation", "t_err": err,
                         "types": "; ".join(v["type"] for v in res["violations"]), "sec": round(time.time() - t1, 1),
                         "notes": ",".join(res.get("notes", []))})
        except Exception as e:
            rows.append({"video": r.video_id, "true": r.violation, "pred": None, "t_err": None, "types": f"ERROR {e}",
                         "sec": round(time.time() - t1, 1), "notes": ""})
    df = pd.DataFrame(rows); df.to_csv(RESULTS / f"tta_{name}.csv", index=False)
    pos, neg = df[df.true], df[~df.true]
    tp, fp = int((pos.pred == True).sum()), int((neg.pred == True).sum())
    print(f"TTA {name}: верно {tp + len(neg) - fp}/{len(df)} | найдено {tp}/{len(pos)} | ложных {fp}/{len(neg)} | "
          f"таймкод медиана {pos.t_err.median():.1f} c | время: медиана {df.sec.median():.0f} c, макс {df.sec.max():.0f} c | "
          f"ошибок {df.pred.isna().sum()}", flush=True)
    print("  пропуски:", ", ".join(pos[pos.pred != True].video), "| ложные:", ", ".join(neg[neg.pred == True].video) or "—", flush=True)
    return df

t0 = time.time()
clean = run("clean", lambda r: files[r.key])
stress = run("stress", lambda r: STRESS / AUG / f"{r.key}.mp4")
print(f"\nбаза: чистые 32/40 (ложных 1), стресс 28/40 (ложных 2) | всего {(time.time() - t0) / 60:.1f} мин")
'''),
]

out = Path(sys.argv[1])
out.write_text(json.dumps(nb(CELLS), indent=1, ensure_ascii=False) + "\n")
print("written", out)

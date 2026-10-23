import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from nb_runpod import nb  # noqa: E402

CELLS = [
    ("md", """
# 10 — Свои видео (настоящий другой ракурс)

Положите ролики в `/workspace/data/own/` (любые: .mov, .mp4). Ноутбук:
1. покажет параметры и первый кадр каждого (видно, если картинка повёрнута);
2. прогонит через тот же `analyze()`, что у жюри (детекторы + правила + Qwen-9B);
3. покажет вердикт, нарушения, таймкоды и кадры-доказательства.

Если знаете, что снимали, впишите в `EXPECTED` — посчитаем совпадение.
"""),
    ("code", r'''
import os, sys, zipfile, time, json, base64, io
from pathlib import Path
import numpy as np, pandas as pd, cv2
import matplotlib.pyplot as plt
from PIL import Image
WORK = Path(os.getenv("WORK", "/workspace")); os.environ.setdefault("HF_HOME", str(WORK / "hf_cache"))
APP, RESULTS = WORK / "app", WORK / "results"; RESULTS.mkdir(exist_ok=True)
zipfile.ZipFile(WORK / "src.zip").extractall(APP); sys.path.insert(0, str(APP)); os.chdir(WORK)
OWN = WORK / "data" / "own"; OWN.mkdir(parents=True, exist_ok=True)

# что снимали: имя файла -> (есть нарушение?, секунда или None, что за нарушение). Можно оставить пустым.
EXPECTED = {
    # "IMG_0001.MOV": (True, 12, "ушла, оставив деньги на столе"),
    # "IMG_0002.MOV": (False, None, "норма"),
}

vids = sorted(p for p in OWN.iterdir() if p.suffix.lower() in (".mov", ".mp4", ".m4v", ".avi", ".mkv"))
print("видео:", len(vids))
fig, axes = plt.subplots(1, max(1, len(vids)), figsize=(4 * max(1, len(vids)), 6))
for ax, p in zip(np.atleast_1d(axes), vids):
    cap = cv2.VideoCapture(str(p)); ok, fr = cap.read()
    fps, n = cap.get(cv2.CAP_PROP_FPS), cap.get(cv2.CAP_PROP_FRAME_COUNT)
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)); cap.release()
    print(f"{p.name}: {w}x{h}, {fps:.0f} fps, {n / max(fps, 1):.1f} c, {p.stat().st_size / 1e6:.0f} МБ")
    if ok: ax.imshow(cv2.cvtColor(fr, cv2.COLOR_BGR2RGB))
    ax.set_title(p.name, fontsize=9); ax.axis("off")
plt.tight_layout(); plt.show()
'''),
    ("code", r'''
from src.pipeline import Analyzer
an = Analyzer(person_weights=str(WORK / "yolo11m.pt"), pose_weights=str(WORK / "yolo11m-pose.pt"),
              object_weights=str(WORK / "models" / "cash_yolo11m_auto.pt"), device="0", vlm_model_id="Qwen/Qwen3.5-9B")
results, rows = {}, []
for p in vids:
    t0 = time.time()
    res = an.analyze(p, job_id=p.stem); results[p.name] = res
    exp = EXPECTED.get(p.name)
    rows.append({"видео": p.name, "вердикт": res["verdict"], "нарушений": len(res["violations"]),
                 "что нашли": "; ".join(f"{v['timecode_sec']:.1f}c {v['type']}" for v in res["violations"]),
                 "ожидали": "" if exp is None else f"{'нарушение' if exp[0] else 'норма'} {exp[1] or ''} {exp[2]}",
                 "совпало": None if exp is None else (res["verdict"] == "violation") == exp[0],
                 "сек": round(time.time() - t0, 1), "заметки": ",".join(res.get("notes", []))})
df = pd.DataFrame(rows); pd.set_option("display.max_colwidth", 120); display(df)
json.dump(results, open(RESULTS / "own_videos_results.json", "w"), ensure_ascii=False)
if df["совпало"].notna().any():
    print(f"совпало с ожиданием: {int(df['совпало'].sum())}/{int(df['совпало'].notna().sum())}")
'''),
    ("md", "## Кадры-доказательства"),
    ("code", r'''
for name, res in results.items():
    for v in res["violations"]:
        fr = v["evidence_frames"]
        fig, axes = plt.subplots(1, len(fr), figsize=(4 * len(fr), 5))
        for ax, f in zip(np.atleast_1d(axes), fr):
            ax.imshow(Image.open(io.BytesIO(base64.b64decode(f["image_base64"])))); ax.set_title(f"{f['timestamp_sec']} c"); ax.axis("off")
        plt.suptitle(f"{name}: {v['type']} @ {v['timecode_sec']} c — {v['description']}", fontsize=10)
        plt.tight_layout(); plt.show()
'''),
]

out = Path(sys.argv[1])
out.write_text(json.dumps(nb(CELLS), indent=1, ensure_ascii=False) + "\n")
print("written", out)

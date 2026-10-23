import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from nb_runpod import nb  # noqa: E402

CELLS = [
    ("md", """
# 06 — Точечные вопросы к Qwen на моментах-кандидатах

Правила по рамкам упёрлись в ~65%: оставшиеся нарушения отличаются **смыслом действия**.
Здесь Qwen отвечает на **узкие вопросы да/нет** только в подозрительные моменты из таймлайнов:

| Вопрос | Где задаём | Ловит |
|---|---|---|
| **count** — пересчитывает купюры руками? | руки рядом с деньгами ≥ 1.5 с | X-1, X-47, X-13, X-44 |
| **outsider** — второй человек на стороне кассира? | в комнате ≥ 2 человек | X-18, X-30 |
| **secure** — перед уходом убрала в сейф / закрыла? | кассир уходит из комнаты | X-11, X-22, X-24, X-28 |

Берём **вероятность «yes»**, пороги подбираем на 40 видео с метками организаторов. `MODEL_ID` можно сменить на `Qwen/Qwen3.5-9B`.
"""),
    ("code", r'''
import os, sys, json, time, zipfile, re, unicodedata
from pathlib import Path
WORK = Path(os.getenv("WORK", "/workspace"))
MODEL_ID = os.getenv("MODEL_ID", "Qwen/Qwen3.5-9B")   # 4B уже проверен: почти не разделяет
DEVICE = os.getenv("DEVICE", "cuda")
os.environ.setdefault("HF_HOME", str(WORK / "hf_cache"))
APP, RESULTS = WORK / "app", WORK / "results"; RESULTS.mkdir(exist_ok=True)
zipfile.ZipFile(WORK / "src.zip").extractall(APP); sys.path.insert(0, str(APP)); os.chdir(WORK)
TIMELINES, VIDEOS_DIR = WORK / "timelines", WORK / "data" / "open" / "videos"
LABELS_XLSX, NOTES_CSV = WORK / "data" / "open" / "open_labels.xlsx", WORK / "data" / "detds" / "video_notes.csv"

import numpy as np, pandas as pd, torch, cv2
from PIL import Image
from IPython.display import display
from src.evaluate import merge_duplicates, score
from src.rules import detect_events
import src.vlm_checks as V

def norm_vid(s):
    s = unicodedata.normalize("NFC", str(s)).strip().replace("Х", "X").replace("х", "X").replace("x", "X")
    return re.sub(r"\.mov$", "", s, flags=re.I)
def parse_ts(v):
    if pd.isna(v): return None
    s = str(v).lower(); m = re.search(r"(\d+)\s*мин", s); c = re.search(r"(\d+)\s*сек", s)
    return float((int(m.group(1)) * 60 if m else 0) + (int(c.group(1)) if c else 0))
lab = pd.read_excel(LABELS_XLSX); lab.columns = ["file", "ts", "label"]
lab["video_id"] = lab.file.map(norm_vid); lab["violation"] = lab.label.str.strip().str.lower().eq("нарушение"); lab["t_sec"] = lab.ts.map(parse_ts)
files = {norm_vid(p.name): p for p in VIDEOS_DIR.glob("*.mov")}; lab["path"] = lab.video_id.map(files)
notes = pd.read_csv(NOTES_CSV); notes["video_id"] = notes["видео"].map(norm_vid)
lab = lab.merge(notes[["video_id", "вероятный_тип"]].rename(columns={"вероятный_тип": "type"}), on="video_id", how="left")
videos = merge_duplicates(lab)
TL = {k: {n: pd.read_parquet(TIMELINES / f"{k}_{n}.parquet") for n in ("persons", "objects", "pose")} for k in videos.key}
cands = {k: V.candidates(d["persons"], d["objects"], d["pose"], 848) for k, d in TL.items()}
print("видео:", len(videos), "| вопросов к VLM:", {q: sum(c["q"] == q for cs in cands.values() for c in cs) for q in V.QUESTIONS})
'''),
    ("md", "## 1. Qwen: вероятность ответа «yes»"),
    ("code", r'''
from transformers import AutoModelForImageTextToText, AutoProcessor
proc = AutoProcessor.from_pretrained(MODEL_ID)
model = AutoModelForImageTextToText.from_pretrained(MODEL_ID, dtype=torch.bfloat16).to(DEVICE).eval()
tok = proc.tokenizer
YES = sorted({tok.encode(w, add_special_tokens=False)[0] for w in ["yes", "Yes", " yes", " Yes", "YES"]})
NO = sorted({tok.encode(w, add_special_tokens=False)[0] for w in ["no", "No", " no", " No", "NO"]})

def grab(path, t, side=640):
    cap = cv2.VideoCapture(str(path)); cap.set(cv2.CAP_PROP_POS_MSEC, max(t, 0) * 1000); ok, fr = cap.read(); cap.release()
    if not ok: return None
    im = Image.fromarray(cv2.cvtColor(fr, cv2.COLOR_BGR2RGB)); im.thumbnail((side, side)); return im

def build(images, question):
    # ответ начинается с "Answer:" -> следующее слово модели ровно yes/no, без "Based on the images..."
    msgs = [{"role": "user", "content": [*({"type": "image", "image": im} for im in images),
                                          {"type": "text", "text": question + " Answer with exactly one word: yes or no."}]},
            {"role": "assistant", "content": [{"type": "text", "text": "Answer:"}]}]
    kw = dict(tokenize=True, return_dict=True, return_tensors="pt", continue_final_message=True)
    try:
        return proc.apply_chat_template(msgs, enable_thinking=False, **kw).to(DEVICE)
    except TypeError:
        return proc.apply_chat_template(msgs, **kw).to(DEVICE)

@torch.no_grad()
def p_yes(images, question):
    inp = build(images, question)
    logits = model(**inp).logits[0, -1].float()
    y, n = torch.logsumexp(logits[YES], 0), torch.logsumexp(logits[NO], 0)
    return float(torch.sigmoid(y - n))

@torch.no_grad()
def answer_text(images, question):
    inp = build(images, question)
    out = model.generate(**inp, max_new_tokens=6, do_sample=False)
    return tok.decode(out[0, inp["input_ids"].shape[1]:], skip_special_tokens=True)

# проверка на известных моментах: ожидание в скобках
checks = [("X-1", 21, "count", "yes"), ("X-18", 20, "outsider", "yes"), ("X-3", 12, "outsider", "no"),
          ("X-24", 12.5, "secure", "no"), ("X-25", 30.1, "secure", "yes")]
for vid, t, q, exp in checks:
    fr = [t, t + 0.75, t + 1.5] if q != "secure" else [t - 9, t - 6, t - 3, t]
    imgs = [im for im in (grab(files[vid], x) for x in fr) if im is not None]
    print(f"{vid} t={t} {q:8s} ожидаем {exp:3s} -> p_yes={p_yes(imgs, V.QUESTIONS[q]):.2f}  текст='{answer_text(imgs, V.QUESTIONS[q]).strip()}'")
'''),
    ("md", "## 2. Все кандидаты (~10–15 мин)"),
    ("code", r'''
rows, t0, done = [], time.time(), 0
total = sum(len(c) for c in cands.values())
for r in videos.itertuples():
    path = files[r.key]
    for c in cands[r.key]:
        imgs = [im for im in (grab(path, x) for x in c["frames"]) if im is not None]
        if imgs:
            rows.append({"video": r.key, "q": c["q"], "t": c["t"], "p_yes": p_yes(imgs, V.QUESTIONS[c["q"]])})
        done += 1
        if done % 50 == 0:
            print(f"{done}/{total}  {(time.time() - t0) / 60:.1f} мин", flush=True)
answers = pd.DataFrame(rows)
OUT_CSV = RESULTS / f"vlm_answers_{MODEL_ID.split('/')[-1]}.csv"
answers.to_csv(OUT_CSV, index=False)
print(f"готово: {len(answers)} ответов за {(time.time() - t0) / 60:.1f} мин -> {OUT_CSV}")
'''),
    ("md", "## 3. Что Qwen ответил: максимум p_yes по видео и вопросу"),
    ("code", r'''
mx = answers.groupby(["video", "q"]).p_yes.max().round(2).unstack("q")
info = videos.set_index("key")[["violation", "t_list", "types"]]
pd.set_option("display.width", 250); pd.set_option("display.max_rows", 60)
display(info.join(mx, how="left").sort_values("violation"))
'''),
    ("md", "## 4. Правила + вопросы Qwen: подбор порогов"),
    ("code", r'''
base = {r.key: detect_events(TL[r.key]["persons"], TL[r.key]["objects"], 848) for r in videos.itertuples()}
by_video = {k: g.to_dict("records") for k, g in answers.groupby("video")}

def run(thr, use=("count", "outsider", "secure")):
    th = {"count": thr.get("count", 2), "outsider": thr.get("outsider", 2), "secure": thr.get("secure", -1)}
    th = {q: (v if q in use else (2 if q != "secure" else -1)) for q, v in th.items()}
    ev = {k: base[k] + V.vlm_events(by_video.get(k, []), th) for k in base}
    return score(videos, ev, 0.5)[0]

print("только правила:", run({}, use=()))
res = []
for q, grid in [("count", [0.5, 0.6, 0.7, 0.8, 0.9]), ("outsider", [0.5, 0.6, 0.7, 0.8, 0.9]), ("secure", [0.1, 0.2, 0.3, 0.4, 0.5])]:
    for v in grid:
        s = run({q: v}, use=(q,)); res.append({"q": q, "thr": v, **{k: s[k] for k in ("recall", "false_alarms", "t_err_median", "type_ok")}})
display(pd.DataFrame(res))
'''),
    ("code", r'''
# лучшая комбинация: впишите пороги, у которых recall растёт, а false_alarms почти нет
BEST = {"count": 0.8, "outsider": 0.8, "secure": 0.3}
s = run(BEST); print("правила + Qwen:", s)
json.dump(BEST, open(APP / "vlm_thresholds.json", "w"))
'''),
]

out = Path(sys.argv[1])
out.write_text(json.dumps(nb(CELLS), indent=1, ensure_ascii=False) + "\n")
print("written", out)

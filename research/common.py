"""Shared helpers for the model benchmark: labels, video probing, frame cache."""
import os
import re
import unicodedata
from pathlib import Path

import av
import numpy as np
import pandas as pd

DATA_DIR = Path(os.getenv("DATA_DIR", Path.home() / "Downloads" / "open"))  # videos/*.mov + open_labels.xlsx (not in repo)
VIDEO_DIR = DATA_DIR / "videos"
LABELS_XLSX = DATA_DIR / "open_labels.xlsx"
ROOT = Path(__file__).resolve().parents[1]
CACHE_DIR = ROOT / "data" / "processed" / "frames"
OUT_DIR = ROOT / "reports" / "bench"


def norm_id(name):
    """'Х-12' (Cyrillic) / 'X-12' (Latin) / 'x-12.mov' -> 'X-12'."""
    s = unicodedata.normalize("NFKC", str(name)).strip()
    s = s.replace("Х", "X").replace("х", "X").replace("x", "X")
    return re.sub(r"\.mov$", "", s, flags=re.I)


def parse_ts(value):
    """'22 секунда' -> 22.0, '1минута 8 секунда' -> 68.0, NaN -> None."""
    if pd.isna(value):
        return None
    s = str(value).lower()
    m = re.search(r"(\d+)\s*мин", s)
    sec = re.search(r"(\d+)\s*сек", s)
    total = (int(m.group(1)) * 60 if m else 0) + (int(sec.group(1)) if sec else 0)
    if not m and not sec:
        total = float(re.search(r"\d+(\.\d+)?", s).group(0))
    return float(total)


def load_labels():
    df = pd.read_excel(LABELS_XLSX)
    df.columns = ["file", "ts", "label"]
    df["video_id"] = df.file.map(norm_id)
    df["violation"] = df.label.str.strip().str.lower().eq("нарушение")
    df["t_sec"] = df.ts.map(parse_ts)
    files = {norm_id(p.name): p for p in VIDEO_DIR.glob("*.mov")}
    df["path"] = df.video_id.map(files)
    missing = df[df.path.isna()]
    assert missing.empty, f"no video for {missing.video_id.tolist()}"
    return df[["video_id", "violation", "t_sec", "path"]].reset_index(drop=True)


def probe(path):
    with av.open(str(path)) as c:
        s = c.streams.video[0]
        dur = float(s.duration * s.time_base) if s.duration else float(c.duration / 1e6)
        return {"codec": s.codec_context.name, "w": s.codec_context.width, "h": s.codec_context.height,
                "fps": float(s.average_rate), "duration": dur, "frames": s.frames,
                "rotation": s.metadata.get("rotate", "")}


def sample_frames(path, fps=2.0, max_side=640):
    """Decode a video and keep ~fps frames per second. Returns (times[N], frames[N,H,W,3] uint8 RGB)."""
    times, frames = [], []
    next_t = 0.0
    with av.open(str(path)) as c:
        s = c.streams.video[0]
        s.thread_type = "AUTO"
        for f in c.decode(s):
            t = float(f.pts * s.time_base)
            if t + 1e-6 < next_t:
                continue
            img = f.to_image()
            scale = max_side / max(img.size)
            if scale < 1:
                img = img.resize((round(img.width * scale), round(img.height * scale)))
            frames.append(np.asarray(img.convert("RGB")))
            times.append(t)
            next_t += 1.0 / fps
    return np.array(times), np.stack(frames)


def cached_frames(video_id, path, fps=2.0, max_side=640):
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    f = CACHE_DIR / f"{video_id}_{fps:g}fps_{max_side}.npz"
    if f.exists():
        z = np.load(f)
        return z["times"], z["frames"]
    times, frames = sample_frames(path, fps, max_side)
    np.savez_compressed(f, times=times, frames=frames)
    return times, frames

"""Probe all videos, build frame cache (2 fps) and contact sheets around labelled timecodes."""
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import OUT_DIR, cached_frames, load_labels, probe  # noqa: E402


def contact_sheet(times, frames, center, span=3.0, cols=7, title=""):
    idx = [i for i, t in enumerate(times) if abs(t - center) <= span]
    tiles = [Image.fromarray(frames[i]).resize((320, int(320 * frames.shape[1] / frames.shape[2]))) for i in idx]
    w, h = tiles[0].size
    rows = int(np.ceil(len(tiles) / cols))
    sheet = Image.new("RGB", (cols * w, rows * (h + 18) + 20), "white")
    d = ImageDraw.Draw(sheet)
    d.text((4, 4), title, fill="black")
    for k, (i, tile) in enumerate(zip(idx, tiles)):
        x, y = (k % cols) * w, 20 + (k // cols) * (h + 18)
        sheet.paste(tile, (x, y))
        d.text((x + 4, y + h + 2), f"t={times[i]:.1f}s", fill="red" if abs(times[i] - center) < 0.6 else "black")
    return sheet


def main():
    labels = load_labels()
    rows = []
    sheets = OUT_DIR / "sheets"
    sheets.mkdir(parents=True, exist_ok=True)
    for r in labels.itertuples():
        t0 = time.perf_counter()
        meta = probe(r.path)
        times, frames = cached_frames(r.video_id, r.path)
        meta.update(video_id=r.video_id, violation=r.violation, t_sec=r.t_sec,
                    decode_s=round(time.perf_counter() - t0, 2), n_cached=len(times))
        rows.append(meta)
        center = r.t_sec if r.violation else times[len(times) // 2]
        contact_sheet(times, frames, center, title=f"{r.video_id} violation={r.violation} t={r.t_sec}") \
            .save(sheets / f"{r.video_id}.jpg", quality=80)
        print(meta, flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(OUT_DIR / "videos.csv", index=False)
    print(df.describe(include="all").T.to_string())


if __name__ == "__main__":
    main()

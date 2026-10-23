"""Atomic-fact check: can a small VLM answer simple yes/no questions about scene state?

Ground truth below was set by eye from the contact sheets (None = not checked).
Usage: python research/vlm_facts.py mlx-community/Qwen3.5-4B-MLX-4bit
"""
import json
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import OUT_DIR, cached_frames, load_labels  # noqa: E402

Q = {
    "cashier_at_desk": "Is the cashier (the woman in pink) at her desk, sitting or standing right next to it?",
    "other_person_in_room": "Is there a person other than the cashier inside this room? Ignore people seen through the glass wall or in reflections.",
    "safe_open": "Is the door of the grey metal safe open?",
    "cash_on_desk": "Are banknotes, a cash pack or a cash bag lying on the desk? Do not count the money counting machine, the laptop or the phone.",
    "phone_in_hand": "Is the cashier holding a mobile phone in her hand?",
}
# (video, t): {fact: truth}
GT = {
    ("X-15", 16.0): {"cashier_at_desk": False, "other_person_in_room": False, "cash_on_desk": True, "phone_in_hand": False},
    ("X-12", 38.0): {"cashier_at_desk": False, "other_person_in_room": False, "cash_on_desk": False, "phone_in_hand": False},
    ("X-13", 49.0): {"safe_open": True, "other_person_in_room": False},
    ("X-3", 12.0): {"cashier_at_desk": True, "other_person_in_room": True, "safe_open": False},
    ("X-17", 7.0): {"cashier_at_desk": True, "other_person_in_room": True, "cash_on_desk": True, "safe_open": False},
    ("X-2", 12.5): {"cashier_at_desk": True, "other_person_in_room": True, "phone_in_hand": True, "safe_open": False},
    ("X-1", 19.0): {"cashier_at_desk": True, "other_person_in_room": True, "phone_in_hand": False, "safe_open": False},
    ("X-4", 54.0): {"cashier_at_desk": True, "other_person_in_room": False, "phone_in_hand": False, "safe_open": False},
}


def main():
    from mlx_vlm import generate, load
    from mlx_vlm.prompt_utils import apply_chat_template
    model_id = sys.argv[1]
    model, processor = load(model_id)
    labels = load_labels().set_index("video_id")
    tmp = Path(tempfile.mkdtemp())
    rows, lat = [], []
    for (vid, t), facts in GT.items():
        times, frames = cached_frames(vid, labels.loc[vid, "path"])
        img = Image.fromarray(frames[int(np.abs(times - t).argmin())])
        p = tmp / f"{vid}.jpg"
        img.save(p, quality=92)
        for fact, truth in facts.items():
            prompt = apply_chat_template(processor, model.config,
                                         f"This is an overhead security camera frame of a bank cash room. {Q[fact]} Answer only yes or no.",
                                         num_images=1)
            t0 = time.perf_counter()
            out = generate(model, processor, prompt, image=[str(p)], max_tokens=5, temperature=0.0, verbose=False)
            lat.append(time.perf_counter() - t0)
            ans = getattr(out, "text", out).strip().lower().startswith("yes")
            rows.append({"video": vid, "t": t, "fact": fact, "truth": truth, "pred": ans, "ok": ans == truth})
    acc = {f: f"{sum(r['ok'] for r in rows if r['fact'] == f)}/{sum(r['fact'] == f for r in rows)}" for f in Q}
    summary = {"model": model_id, "accuracy_by_fact": acc,
               "total": f"{sum(r['ok'] for r in rows)}/{len(rows)}",
               "s_per_question": round(float(np.median(lat)), 2),
               "errors": [r for r in rows if not r["ok"]]}
    (OUT_DIR / f"facts_{model_id.replace('/', '__')}.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

#!/bin/zsh
# Sequential VLM benchmark suite (one model in memory at a time).
cd "$(dirname "$0")/.."
PY=".venv/bin/python -I research/vlm_bench.py"
F="grep -v -E 'objc|Redirects|Fetching|it/s'"
eval "$PY mlx-community/Qwen3.5-4B-MLX-4bit --stride 3 --tag all 2>&1 | $F | grep -E 'proc_s|video_auc|recall|Error|Traceback'"
eval "$PY mlx-community/Qwen3.5-2B-MLX-4bit --stride 3 --tag all 2>&1 | $F | grep -E 'proc_s|video_auc|recall|Error|Traceback'"
eval "$PY mlx-community/Qwen3.5-4B-MLX-4bit --stride 3 --prompt no_entry --videos X-2,X-4,X-17,X-3,X-5 --tag unknown 2>&1 | $F | tail -30"
for seed in 1 2 3; do
  eval "$PY mlx-community/Qwen3.5-4B-MLX-4bit --stride 3 --temp 0.7 --seed $seed --videos X-1,X-11,X-15,X-3,X-12 --tag seed$seed 2>&1 | $F | tail -30"
done
echo SUITE_DONE

#!/bin/bash
# Проверка API как это сделает жюри: bash /workspace/test_api.sh [видео] [адрес]
VIDEO="${1:-$(ls /workspace/data/open/videos/*-18.mov | head -1)}"
BASE="${2:-https://${RUNPOD_POD_ID}-8000.proxy.runpod.net}"
KEY="$(cat /workspace/service/api_key.txt)"
echo "без ключа (ждём 401):"; curl -s -o /dev/null -w "%{http_code}\n" -X POST "$BASE/api/v1/analyze" -F "file=@$VIDEO"
T0=$(date +%s)
JOB=$(curl -s -X POST "$BASE/api/v1/analyze" -H "X-API-Key: $KEY" -F "file=@$VIDEO" | python -c "import sys, json; print(json.load(sys.stdin)['job_id'])")
echo "job_id=$JOB"
while true; do
  R=$(curl -s "$BASE/api/v1/jobs/$JOB" -H "X-API-Key: $KEY")
  S=$(echo "$R" | python -c "import sys, json; print(json.load(sys.stdin)['status'])")
  [ "$S" != "processing" ] && break
  sleep 3
done
echo "статус=$S за $(( $(date +%s) - T0 )) c"
echo "$R" | python -c "
import sys, json
d = json.load(sys.stdin)
print('verdict:', d.get('verdict'))
for v in d.get('violations', []):
    print(f\"  {v['timecode_sec']:6.1f} c  {v['type']}  кадров={len(v['evidence_frames'])}\")
"

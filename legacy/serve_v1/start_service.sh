#!/bin/bash
# Запуск сервиса для жюри на поде RunPod (порт 8000).
#   bash /workspace/start_service.sh            -> старт (ключ создаётся один раз и хранится в /workspace/service/api_key.txt)
#   FORCE=1 bash /workspace/start_service.sh    -> остановить Label Studio / старый сервис на порту 8000 и стартовать
set -e
cd /workspace
mkdir -p /workspace/service /workspace/app
python -m zipfile -e /workspace/service.zip /workspace/app >/dev/null
pip install -q fastapi uvicorn python-multipart

if pgrep -f "label-studio start" >/dev/null || pgrep -f "uvicorn serve.app:app" >/dev/null; then
  if [ "$FORCE" != "1" ]; then
    echo "Порт 8000 занят (Label Studio или старый сервис). Если разметка закончена: FORCE=1 bash /workspace/start_service.sh"
    exit 1
  fi
  pkill -f "label-studio start" || true
  pkill -f "uvicorn serve.app:app" || true
  sleep 2
fi

KEY_FILE=/workspace/service/api_key.txt
[ -s "$KEY_FILE" ] || python -c "import secrets; print(secrets.token_urlsafe(24))" > "$KEY_FILE"
export API_KEY="$(cat $KEY_FILE)"
export WORK=/workspace HF_HOME=/workspace/hf_cache
export OBJ_WEIGHTS=/workspace/models/cash_yolo11m_auto.pt
export VLM_MODEL_ID="${VLM_MODEL_ID-Qwen/Qwen3.5-9B}"

nohup uvicorn serve.app:app --app-dir /workspace/app --host 0.0.0.0 --port 8000 > /workspace/service/service.log 2>&1 &
echo "Сервис запускается, грузятся модели (1-3 мин)..."
for i in $(seq 1 90); do
  sleep 4
  H=$(curl -s localhost:8000/api/v1/health || true)
  if echo "$H" | grep -q '"ready":true'; then echo "ГОТОВ: $H"; break; fi
  if echo "$H" | grep -q '"error":"'; then echo "ОШИБКА: $H"; tail -30 /workspace/service/service.log; exit 1; fi
done
POD=${RUNPOD_POD_ID:-<POD_ID>}
echo "Страница:  https://${POD}-8000.proxy.runpod.net/"
echo "API:       https://${POD}-8000.proxy.runpod.net/api/v1/analyze"
echo "API-ключ лежит в $KEY_FILE (cat $KEY_FILE)"
echo "Лог:       tail -f /workspace/service/service.log"

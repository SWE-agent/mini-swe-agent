#!/usr/bin/env bash
set -u
cd /data/jiahao/code/mini-swe-agent
source /home/jiahao/miniconda3/etc/profile.d/conda.sh
conda activate swe_agent
export LITELLM_MODEL_REGISTRY_PATH=/data/jiahao/code/mini-swe-agent/experiments/registry.json
OUT=experiments/qwen3_baseline_verified
LOG=experiments/logs/verified.log
while true; do
  echo "[$(date -Iseconds)] Starting/resuming swebench run..." | tee -a "$LOG"
  mini-extra swebench --subset verified --split test \
    -c swebench -c experiments/qwen3_coder_baseline.yaml \
    --workers 4 \
    --output "$OUT" \
    2>&1 | tee -a "$LOG"
  rc=${PIPESTATUS[0]}
  completed=$(python - <<'PY' 2>/dev/null || echo 0
import json, os
p = "experiments/qwen3_baseline_verified/preds.json"
try:
    d = json.load(open(p))
    print(len(d))
except Exception:
    print(0)
PY
)
  echo "[$(date -Iseconds)] exit=$rc completed=$completed/500" | tee -a "$LOG"
  if [ "$completed" -ge 500 ]; then
    echo "[$(date -Iseconds)] DONE" | tee -a "$LOG"
    break
  fi
  echo "[$(date -Iseconds)] sleeping 60s before resume..." | tee -a "$LOG"
  sleep 60
done

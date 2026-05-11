#!/usr/bin/env bash
# Rerun for the 195 empty-patch instances on the cleaned-up docker daemon.
# Polite to other tenants: nice, ionice, ulimit, workers=3.
set -euo pipefail

cd /data/jiahao/code/mini-swe-agent
source /home/jiahao/miniconda3/etc/profile.d/conda.sh
conda activate swe_agent

export LITELLM_MODEL_REGISTRY_PATH=/data/jiahao/code/mini-swe-agent/experiments/registry.json
OUT=experiments/qwen3_baseline_verified
LOG=experiments/logs/rerun_empties.log
mkdir -p "$(dirname "$LOG")"

# Cap this process tree's memory footprint (virtual): 64 GB.
# mini-swe-agent itself is tiny; this bounds any runaway subprocess.
ulimit -v 67108864 || true
# Cap open files reasonably (docker exec + trajectory writers can eat FDs)
ulimit -n 8192 || true

# Polite CPU + I/O scheduling
exec nice -n 19 ionice -c 3 \
  mini-extra swebench --subset verified --split test \
  -c swebench -c experiments/qwen3_coder_baseline.yaml \
  --workers 3 \
  --output "$OUT" 2>&1 | tee -a "$LOG"

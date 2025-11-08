#!/bin/bash
# Evaluate memory version results on SWE-bench

set -e

source /common/users/wx139/env/minisweagentenv/bin/activate
cd /common/users/wx139/code/mini-swe-agent

echo "Converting memory predictions to JSONL..."
python experiments/convert_to_jsonl.py \
    experiments/results/memory/preds.json \
    experiments/results/memory/all_preds.jsonl

echo "Running SWE-bench evaluation..."
python -m swebench.harness.run_evaluation \
    --dataset_name princeton-nlp/SWE-bench_Lite \
    --predictions_path experiments/results/memory/all_preds.jsonl \
    --max_workers 4 \
    --run_id memory_eval

echo "Done!"

#!/usr/bin/env bash
# Submit predictions to sb-cli and report whether resolved >= 30%.
# Runs after the full 500 batch completes.
#
# Usage: bash experiments/submit_and_check.sh <preds_dir> <run_id>
set -euo pipefail

PREDS_DIR="${1:-experiments/outputs/planmem_full}"
RUN_ID="${2:-planmem_qwen3coder_$(date +%Y%m%d_%H%M)}"
PREDS_PATH="${PREDS_DIR}/preds.json"

if [ ! -f "${PREDS_PATH}" ]; then
  echo "ERROR: ${PREDS_PATH} missing" >&2
  exit 2
fi

count=$(jq 'length' "${PREDS_PATH}")
empty=$(jq '[.[] | select(.model_patch == "")] | length' "${PREDS_PATH}")
echo "preds count: ${count}"
echo "empty patches: ${empty}"

echo
echo "=== submitting to sb-cli (run_id=${RUN_ID}) ==="
sb-cli submit swe-bench_verified test \
    --predictions_path "${PREDS_PATH}" \
    --run_id "${RUN_ID}" 2>&1 | tee "${PREDS_DIR}/sb_cli_submit.log"

echo
echo "=== sb-cli will email/poll the report — checking for results ==="
# Poll for the report (max ~30 min)
report_path="sb-cli-reports/swe-bench_verified__test__${RUN_ID}.json"
mkdir -p sb-cli-reports
start=$(date +%s)
while true; do
  if sb-cli get-report swe-bench_verified test "${RUN_ID}" \
       --output_dir sb-cli-reports 2>>"${PREDS_DIR}/sb_cli_get.log"; then
    if [ -f "${report_path}" ]; then
      break
    fi
  fi
  now=$(date +%s)
  elapsed=$(( now - start ))
  if [ "${elapsed}" -gt 1800 ]; then
    echo "TIMEOUT waiting for sb-cli report after 30 min" >&2
    exit 3
  fi
  sleep 60
done

echo
echo "=== report ==="
resolved=$(jq '.resolved // .resolved_instances // 0' "${report_path}")
total=$(jq '.completed // .total_instances // 0' "${report_path}")
unresolved=$(jq '.unresolved // 0' "${report_path}")
pending=$(jq '.pending // 0' "${report_path}")
failed=$(jq '.failed // 0' "${report_path}")
echo "resolved   = ${resolved}"
echo "unresolved = ${unresolved}"
echo "pending    = ${pending}"
echo "failed     = ${failed}"
echo "completed  = ${total}"
if [ "${total}" -gt 0 ]; then
  pct=$(echo "scale=2; ${resolved} * 100 / ${total}" | bc)
  echo "resolved%  = ${pct}%"
  threshold=$(echo "${pct} >= 30" | bc)
  if [ "${threshold}" = "1" ]; then
    echo "STATUS: PASS (>= 30%)"
    exit 0
  else
    echo "STATUS: FAIL (< 30%)"
    exit 1
  fi
fi
echo "STATUS: UNKNOWN (no completed)"
exit 1

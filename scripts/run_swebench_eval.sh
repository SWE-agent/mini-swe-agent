#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

OUTPUT_DIR="output/deepseek_lite_dev"
DATASET_NAME=""
SPLIT=""
RUN_ID=""
MAX_WORKERS="${MAX_WORKERS:-4}"
TIMEOUT="${TIMEOUT:-1800}"
NAMESPACE="${NAMESPACE:-swebench}"
CACHE_LEVEL="${CACHE_LEVEL:-env}"
CLEAN="${CLEAN:-false}"
FORCE_REBUILD="${FORCE_REBUILD:-false}"
EXTRA_ARGS=()

usage() {
  cat <<'EOF'
Usage:
  bash scripts/run_swebench_eval.sh [options]

Options:
  -o, --output-dir PATH      mini-swe-agent output dir containing preds.json
  -d, --dataset-name NAME    SWE-bench dataset name
  -s, --split NAME           Dataset split
  -r, --run-id ID            Evaluation run_id written under logs/run_evaluation/
  -w, --max-workers N        Number of parallel evaluation workers
  -t, --timeout SEC          Per-instance timeout in seconds
  -n, --namespace NAME       Docker namespace passed to swebench harness
      --cache-level LEVEL    One of: none, base, env, instance
      --clean BOOL           Whether to clean images above cache level
      --force-rebuild BOOL   Whether to rebuild images before evaluation
  -h, --help                 Show this help

Examples:
  bash scripts/run_swebench_eval.sh
  bash scripts/run_swebench_eval.sh -o output/deepseek_lite_dev
  bash scripts/run_swebench_eval.sh -o output/deepseek_bench1 -d princeton-nlp/SWE-Bench_Verified -s test

Notes:
  - This script is meant to run inside WSL/Linux.
  - It prefers .wsl-eval-venv/bin/python when available.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    -o|--output-dir)
      OUTPUT_DIR="$2"
      shift 2
      ;;
    -d|--dataset-name)
      DATASET_NAME="$2"
      shift 2
      ;;
    -s|--split)
      SPLIT="$2"
      shift 2
      ;;
    -r|--run-id)
      RUN_ID="$2"
      shift 2
      ;;
    -w|--max-workers)
      MAX_WORKERS="$2"
      shift 2
      ;;
    -t|--timeout)
      TIMEOUT="$2"
      shift 2
      ;;
    -n|--namespace)
      NAMESPACE="$2"
      shift 2
      ;;
    --cache-level)
      CACHE_LEVEL="$2"
      shift 2
      ;;
    --clean)
      CLEAN="$2"
      shift 2
      ;;
    --force-rebuild)
      FORCE_REBUILD="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    --)
      shift
      EXTRA_ARGS+=("$@")
      break
      ;;
    *)
      echo "Unknown argument: $1" >&2
      echo >&2
      usage >&2
      exit 1
      ;;
  esac
done

OUTPUT_BASENAME="$(basename "${OUTPUT_DIR}")"

if [[ -z "${DATASET_NAME}" ]]; then
  if [[ "${OUTPUT_BASENAME}" == *lite* ]]; then
    DATASET_NAME="princeton-nlp/SWE-Bench_Lite"
  else
    DATASET_NAME="princeton-nlp/SWE-Bench_Verified"
  fi
fi

if [[ -z "${SPLIT}" ]]; then
  if [[ "${OUTPUT_BASENAME}" == *_dev ]]; then
    SPLIT="dev"
  elif [[ "${OUTPUT_BASENAME}" == *_test ]]; then
    SPLIT="test"
  else
    SPLIT="test"
  fi
fi

if [[ -z "${RUN_ID}" ]]; then
  RUN_ID="${OUTPUT_BASENAME}_eval"
fi

PREDICTIONS_PATH="${REPO_ROOT}/${OUTPUT_DIR}/preds.json"

if [[ ! -f "${PREDICTIONS_PATH}" ]]; then
  echo "Predictions file not found: ${PREDICTIONS_PATH}" >&2
  exit 1
fi

if [[ -x "${REPO_ROOT}/.wsl-eval-venv/bin/python" ]]; then
  PYTHON_BIN="${REPO_ROOT}/.wsl-eval-venv/bin/python"
elif [[ -x "${REPO_ROOT}/.venv/bin/python" ]]; then
  PYTHON_BIN="${REPO_ROOT}/.venv/bin/python"
elif command -v python3 >/dev/null 2>&1; then
  PYTHON_BIN="$(command -v python3)"
else
  echo "No Linux python executable found. Expected .wsl-eval-venv/bin/python or python3." >&2
  exit 1
fi

cd "${REPO_ROOT}"

echo "Repo root       : ${REPO_ROOT}"
echo "Python          : ${PYTHON_BIN}"
echo "Predictions     : ${PREDICTIONS_PATH}"
echo "Dataset         : ${DATASET_NAME}"
echo "Split           : ${SPLIT}"
echo "Run ID          : ${RUN_ID}"
echo "Max workers     : ${MAX_WORKERS}"
echo "Timeout         : ${TIMEOUT}"
echo "Namespace       : ${NAMESPACE}"
echo "Cache level     : ${CACHE_LEVEL}"
echo "Clean           : ${CLEAN}"
echo "Force rebuild   : ${FORCE_REBUILD}"

exec "${PYTHON_BIN}" -m swebench.harness.run_evaluation \
  --dataset_name "${DATASET_NAME}" \
  --split "${SPLIT}" \
  --predictions_path "${PREDICTIONS_PATH}" \
  --max_workers "${MAX_WORKERS}" \
  --timeout "${TIMEOUT}" \
  --cache_level "${CACHE_LEVEL}" \
  --clean "${CLEAN}" \
  --force_rebuild "${FORCE_REBUILD}" \
  --run_id "${RUN_ID}" \
  --namespace "${NAMESPACE}" \
  "${EXTRA_ARGS[@]}"

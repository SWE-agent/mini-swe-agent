# PlanMem on SWE-bench Verified — runbook

mini-swe-agent v2.2.8 + Qwen3-Coder-30B-A3B-Instruct-FP8 served via vLLM.

## 0. Prereqs

- `docker` running (SWE-bench evaluation containers are x86_64 Linux)
- 2× GPU for vLLM tensor-parallel (or 1× with `--max-model-len 32768` if memory tight)
- `uv` installed; this repo with `uv sync` once

## 1. Serve the model with vLLM (single GPU, e.g. GPU3)

```bash
conda activate vllm_qwen
HF_HOME=/data/jiahao/.cache/huggingface \
TRITON_CACHE_DIR=/data/jiahao/.cache/triton \
VLLM_CACHE_ROOT=/data/jiahao/.cache/vllm \
CUDA_VISIBLE_DEVICES=3 \
vllm serve Qwen/Qwen3-Coder-30B-A3B-Instruct-FP8 \
  --gpu-memory-utilization 0.85 \
  --max-model-len 32768 \
  --host 0.0.0.0 --port 8765 \
  --served-model-name Qwen/Qwen3-Coder-30B-A3B-Instruct-FP8 \
  --enable-auto-tool-choice \
  --tool-call-parser qwen3_coder
```

For 2 GPUs (e.g. 0,3) bump max-model-len and add `--tensor-parallel-size 2`.

Confirm with: `curl http://localhost:8765/v1/models`

## 2. Single-instance smoke (do this first!)

```bash
uv run python -m experiments.smoke_planmem_swebench \
    --instance astropy__astropy-12907 \
    --model openai/Qwen/Qwen3-Coder-30B-A3B-Instruct-FP8 \
    --api-base http://localhost:8765/v1 \
    --out experiments/outputs/planmem_smoke
```

Expected output ends with:
```
exit_status   : Submitted
n_calls       : <int>
cost          : $<float>
wall time     : ~minutes
planner       : N/M sub-tasks done, K failed, phase=...
nodes w/ filename metadata: <int>      ← MUST be > 0
```

If `exit_status != Submitted` or `nodes w/ filename metadata == 0`, **STOP**
and inspect `experiments/outputs/planmem_smoke/<instance>.traj.json`. The
likely culprits:
- vLLM not reachable / wrong tool-call parser
- Docker image pull failing for that instance
- toolcall-mode `extra.actions` shape mismatch (we'd see metadata=0)

## 3. Small batch (5 instances, ablation)

Once the smoke passes:

```bash
uv run python -m minisweagent.run.benchmarks.swebench_planmem \
    --subset verified --split test \
    --slice 0:5 \
    --workers 2 \
    -c src/minisweagent/config/benchmarks/swebench_planmem.yaml \
    -c model.api_base=http://localhost:8765/v1 \
    -m openai/Qwen/Qwen3-Coder-30B-A3B-Instruct-FP8 \
    --model-class litellm \
    -o experiments/outputs/planmem_5
```

Outputs:
- `experiments/outputs/planmem_5/preds.json`
- `experiments/outputs/planmem_5/<instance_id>/<instance_id>.traj.json`
- `experiments/outputs/planmem_5/minisweagent.log`

## 4. Full SWE-bench Verified

```bash
uv run python -m minisweagent.run.benchmarks.swebench_planmem \
    --subset verified --split test \
    --workers 8 \
    -c src/minisweagent/config/benchmarks/swebench_planmem.yaml \
    -c model.api_base=http://localhost:8765/v1 \
    -m openai/Qwen/Qwen3-Coder-30B-A3B-Instruct-FP8 \
    --model-class litellm \
    -o experiments/outputs/planmem_full
```

`workers` ≤ vLLM's effective concurrency. Qwen3-Coder-30B FP8 on 2× GPUs
typically saturates around 4-8 concurrent requests.

## 5. Submit to sb-cli

```bash
sb-cli submit swe-bench_verified test \
    --predictions_path experiments/outputs/planmem_full/preds.json \
    --run_id planmem_qwen3_coder_$(date +%Y%m%d)
```

## 6. Ablation

Use [experiments/run_ablation.py](run_ablation.py) — same model + base config,
toggling feature flags per variant.

```bash
uv run python -m experiments.run_ablation \
    --instances 50 \
    --variants baseline,full \
    -m openai/Qwen/Qwen3-Coder-30B-A3B-Instruct-FP8 \
    --config src/minisweagent/config/benchmarks/swebench_planmem.yaml
```

# Project notes for Claude

## GPU / resource hygiene

When running anything that touches GPUs (vLLM, training, multi-worker SWE-bench
batches, eval scripts, etc.):

- **Use `nice`** to lower scheduling priority for long-running jobs so other
  users on the box are not starved:
  ```bash
  nice -n 10 uv run python -m minisweagent.run.benchmarks.swebench_planmem ...
  ```
- **Use `setrlimit` / `ulimit`** to cap memory before launching, so a runaway
  process cannot OOM the whole machine:
  ```bash
  # 64 GB cap (adjust per host)
  ulimit -v $((64 * 1024 * 1024))   # virtual memory in KB
  nice -n 10 uv run python ...
  ```
  In Python helpers: `import resource; resource.setrlimit(resource.RLIMIT_AS, (64*2**30, 64*2**30))`
- **Pin GPUs explicitly** via `CUDA_VISIBLE_DEVICES=<id>` — never let a job grab
  whichever GPU happens to be free; we share the box.
- **vLLM**: cap `--gpu-memory-utilization` (e.g. `0.85`) and `--max-model-len`
  (e.g. `32768`) instead of letting it consume the whole VRAM pool.
- **Workers**: keep `--workers` ≤ what the served model can actually handle.
  For Qwen3-Coder-30B FP8 on a single L40S, `--workers 4` is safe.

If a script ignores these guardrails, prefer fixing the script over silently
running it without limits.

## Where things live

- PlanMem agent: `src/minisweagent/agents/planmem_agent.py`, `planmem/`, `memory_search.py`
- Tests: `tests/agents/test_planmem.py`
- SWE-bench runner (PlanMem): `src/minisweagent/run/benchmarks/swebench_planmem.py`
- Default config (PlanMem): `src/minisweagent/config/benchmarks/swebench_planmem.yaml`
- Smoke / runbook: `experiments/smoke_planmem_swebench.py`, `experiments/RUN_PLANMEM.md`
- Recall@k harness: `experiments/eval/memory_recall.py`
- Ablation runner: `experiments/run_ablation.py`

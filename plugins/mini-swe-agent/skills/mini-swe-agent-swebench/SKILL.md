---
name: mini-swe-agent-swebench
description: >-
  Run mini-SWE-agent on SWE-bench (or any SWE-bench-format dataset) in batch or
  single-instance mode, pick an isolated environment class, and evaluate the
  resulting preds.json with sb-cli or the local harness. Use when benchmarking
  a model or config, or debugging a single SWE-bench instance.
---

# SWE-bench with mini-SWE-agent

Reference: `docs/usage/swebench.md`, `docs/advanced/environments.md`.

## Batch

```bash
mini-extra swebench \
  --model anthropic/claude-sonnet-4-5-20250929 \
  --subset verified --split test \
  --workers 4 -o runs/verified-sonnet
```

- Data: `--subset` (default `lite`, or a path to a dataset loadable with
  `datasets.load_dataset(path, split=split)`), `--split` (default `dev`),
  `--slice 0:5`, `--filter <regex>`, `--shuffle`, `--redo-existing`.
- Config: `-c` (default `swebench.yaml`).
- Writes `preds.json` and per-instance trajectories into `-o`.

## Single instance (debugging)

```bash
mini-extra swebench-single --subset verified --split test \
  -m anthropic/claude-sonnet-4-5-20250929 -i sympy__sympy-15599
```

`-i` takes an instance ID or index. Add `--exit-immediately` to skip the
confirmation at exit. It does not write `preds.json`.

## Environment class

Always run SWE-bench isolated: `--environment-class docker` (default
recommendation) or `singularity` on HPC without Docker. Others:
`swerex_docker`, `swerex_modal` (Modal cloud), `bubblewrap` (Linux,
experimental), `contree`. `local` has no isolation.

The containers assume x86 Linux. Slow first starts are image pulls; raise
`environment.pull_timeout` (default 120 s) if pulls time out.

## Evaluate

```bash
# Cloud (free, fast)
sb-cli submit swe-bench_verified test --predictions_path preds.json --run_id my-run

# Local harness
python -m swebench.harness.run_evaluation \
  --dataset_name princeton-nlp/SWE-bench_Verified \
  --predictions_path preds.jsonl --max_workers 8 --run_id my-run
```

## Gotchas

- Completed instances are read from `preds.json`; delete entries there (not
  just trajectories) to rerun them.
- Aborted runs can leave `KeyboardInterrupt` entries in `preds.json`.
- Set `MSWEA_GLOBAL_COST_LIMIT` before large runs.
- `run.env_startup_command` runs a command in the environment before the
  agent starts.

# RepoPilot

**Execution-grounded SWE agent evaluation** built on [mini-swe-agent](https://github.com/SWE-agent/mini-swe-agent).

RepoPilot is not primarily a new coding agent or multi-agent demo. It is a **trace-based evaluation harness** that runs benchmark tasks, records structured trajectories, attributes failures from pytest output, and produces aggregate comparison reports—so you can answer *why* an agent succeeded or failed, *at which step*, *on which files*, and *at what cost*.

Upstream `mini` remains the baseline agent. RepoPilot adds orchestration, trace recording, and eval on top.

## What you get

| Layer | Status | Description |
|-------|--------|-------------|
| **Run** | Done | Reproducible benchmark runner with git worktree isolation |
| **Trace** | Done (v2) | `trace.json`, failure attribution, patch/test extraction |
| **Eval** | Done | Summary, compare, breakdown by tags and failure taxonomy |
| **Visualize** | Planned | Trajectory charts and HTML reports (Phase 5) |
| **Self-repair / roles** | Deferred | Optional experiments, eval-gated |

## Project layout

```text
RepoPilot-SWE-Multi-agent/
├── upstream/              # Frozen mini-swe-agent (baseline reference)
├── src/repopilot/         # Runner, schema, trace, eval
├── benchmarks/            # 14 tagged benchmark tasks (task_001–014)
├── tests/repopilot/       # RepoPilot unit tests
├── scripts/               # Patch verification, baseline extraction
├── runs/                  # Run outputs (gitignored)
```

See [benchmarks/README.md](benchmarks/README.md) for task definitions.

## Install

```bash
python3 -m venv venv && source venv/bin/activate

# Upstream mini CLI first, then RepoPilot
pip install -e 'upstream/[dev]'
pip install -e '.[dev]'
```

Configure model access (same as upstream mini):

```bash
export OPENAI_API_KEY=...
# optional: export OPENAI_BASE_URL=... MSWEA_MODEL_NAME=...
```

Verify:

```bash
mini --help
repopilot --help
python -m pytest tests/repopilot -q
```

## Quick start

**Run a benchmark task** (uses upstream `mini` in baseline mode):

```bash
repopilot run task_001_sudoku
```

**Dry-run** (print planned commands only):

```bash
repopilot run task_001_sudoku --dry-run
```

**Verify patch + test setup without calling the LLM:**

```bash
repopilot run task_001_sudoku --skip-mini
python scripts/verify_benchmark_patches.py
```

**Re-record trace artifacts** from an existing trajectory:

```bash
repopilot trace runs/task_001_sudoku/trajectory.traj.json
```

**Aggregate eval report** from all runs under `runs/`:

```bash
repopilot eval summary
# → runs/eval/summary/eval_report.md, metrics.json, failure_breakdown.md, ...
```

## Benchmarks

14 reproducible tasks with tagged failure modes:

| Tier | Tasks | Notes |
|------|-------|-------|
| Single-bug | `task_001`–`task_010` | off-by-one, wrong operator, import path, logic, etc. |
| Multi-bug | `task_011` (3), `task_012` (4), `task_013`/`task_014` (5) | `eval.bug_count` + `failure_mode: composite` |

Each task directory contains `config.yaml`, `issue.md`, and optional `setup.patch` (plants a known bug at `base_commit`).

Run outputs per task:

```text
runs/{task_id}/
  trajectory.traj.json   # upstream mini trajectory
  trace.json             # structured trace (schema v2)
  patch.diff             # extracted patch
  test.log               # pytest output
  final_report.md        # human-readable summary
  failure_reason.md      # pytest-grounded failure (if failed)
  run_meta.yaml          # task config snapshot
```

## CLI reference

```bash
repopilot run <task_id> [--dry-run] [--skip-mini] [--no-restore]
repopilot trace <trajectory.traj.json> [-o output_dir]

repopilot eval summary [--runs-dir runs]
repopilot eval compare [--task task_id] [--runs-dir runs]
repopilot eval breakdown [--by failure_mode|difficulty|bug_count|failure_category|failure_stage]
```

`repopilot-baseline` is an alias entry point with the same commands.

## Trace & eval highlights

Every run with trace v2 records:

- Step timeline with inferred stages (`read`, `edit`, `test`, `submit`, …)
- Tool calls, files touched, pytest runs before/after edits
- Failure fields: `failure_category`, `failure_stage`, `failed_step`, `failure_message`
- Task tags: `failure_mode`, `difficulty`, `bug_count` (multi-bug tasks)

Eval reports answer six audit questions: outcome, failure reason, failing step, repair effectiveness, cost, and baseline comparison.

## Development

```bash
# RepoPilot tests only
python -m pytest tests/repopilot -q

# Verify all benchmark patches apply and fail tests as expected
python scripts/verify_benchmark_patches.py

# Upstream mini tests (optional)
cd upstream && python -m pytest tests -q
```

**Style:** Python 3.10+, type hints, `pathlib`, `typer`, `pytest`. Keep changes minimal and eval-first—new agent features should be justified by measurable eval improvement.

## Upstream

The baseline agent lives in [`upstream/`](upstream/) (mini-swe-agent v2.4.2). See [upstream/README.md](upstream/README.md) and [mini-swe-agent docs](https://mini-swe-agent.com/latest/) for agent internals, model config, and sandbox options.

## References

- [benchmarks/README.md](benchmarks/README.md) — task catalog and schema
- [SWE-agent paper](https://arxiv.org/abs/2405.15793) — upstream attribution

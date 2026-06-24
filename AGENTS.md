# RepoPilot agent guide

Guidance for AI assistants working in this repository.

## What this repo is

**RepoPilot** extends [mini-swe-agent](https://github.com/SWE-agent/mini-swe-agent) with a **trace-based evaluation harness**. The primary artifact is measurable, comparable benchmark runs—not a multi-agent demo.

- **Upstream agent:** `upstream/` — frozen mini-swe-agent; baseline runs call `mini` via subprocess.
- **RepoPilot code:** `src/repopilot/` — schema, runner, trace recorder, eval harness.
- **Benchmarks:** `benchmarks/task_*` — YAML + issue + setup.patch; 14 tasks (single- and multi-bug).
- **Outputs:** `runs/` — gitignored; never commit unless the user asks.

Read [README.md](README.md) for install/CLI.

## Architecture

```text
Run (runner)  →  Trace (recorder)  →  Eval (loader/metrics/report)  →  Agent experiments (deferred)
```

**Eval-first rule:** extend trace schema and eval metrics before adding agent features (self-repair, role splits). New agent behavior must be justified by eval comparison.

## Project structure

```text
src/repopilot/
  schema/       TaskConfig (Pydantic), discover_tasks, load_task
  runner/       run_task.py, benchmark.py (CLI), baseline.py
  trace/        parse, classify, recorder (trace v2)
  eval/         loader, metrics, report, compare, failure_analysis, trajectory_analysis

benchmarks/task_XXX_name/
  config.yaml   task_id must match directory name
  issue.md      prompt for mini
  setup.patch   optional bug injection at base_commit

tests/repopilot/   RepoPilot tests only (upstream tests live in upstream/tests/)
```

## Common commands

```bash
pip install -e 'upstream/[dev]' && pip install -e '.[dev]'

repopilot run task_001_sudoku
repopilot run task_011_sudoku_multi3 --skip-mini
repopilot eval summary
repopilot eval breakdown --by bug_count
python -m pytest tests/repopilot -q
python scripts/verify_benchmark_patches.py
```

## Coding conventions

Inherited from upstream mini-swe-agent, adapted for RepoPilot:

1. Target Python 3.10+; use `list` not `List`, `pathlib` not `os.path`.
2. Use `typer` for CLI; `pydantic` v2 for config models.
3. Keep code minimal—prefer focused diffs over abstractions.
4. Do not catch exceptions unless explicitly required.
5. Comments only for non-obvious logic.
6. Tests: `pytest` only; avoid mocking unless asked; no trivial assertions.
7. `pytest.mark.parametrize` first arg must be a **tuple**, second a **list**.

## Where to change what

| Task | Location |
|------|----------|
| Add benchmark task | `benchmarks/task_XXX/`, update `benchmarks/README.md`, `tests/repopilot/test_task_schema.py`, `scripts/verify_benchmark_patches.py` |
| Run lifecycle / worktree | `src/repopilot/runner/run_task.py` |
| CLI commands | `src/repopilot/runner/benchmark.py` |
| Trace schema / failure taxonomy | `src/repopilot/trace/` |
| Eval aggregation / reports | `src/repopilot/eval/` |
| Task config fields | `src/repopilot/schema/task.py` + wire through `recorder.py`, `loader.py`, `run_task.py` |

## Benchmark task rules

- `task_id` in `config.yaml` must match the directory name.
- `repo.base_commit` is checked out in an isolated git worktree under `runs/{task_id}/.workspace`.
- `setup.patch` must apply cleanly and cause pytest to **fail** before the agent runs.
- Tag tasks with `eval.failure_mode`, `eval.difficulty`; multi-bug tasks use `eval.bug_count` and `tags: [multi_bug]`.
- Do not edit upstream source for benchmarks—use `setup.patch` to inject bugs.

## Git & commits

- Only commit when the user explicitly asks.
- Do not commit `runs/`, `.env`, or credentials.
- Prefer conventional prefixes: `feat(eval):`, `fix(trace):`, `docs:`, etc.
- Do **not** add `Co-authored-by: Cursor` trailers.

## Phase status (summary)

| Phase | Status |
|-------|--------|
| 0–1 | Done — baseline, schema, runner |
| 2–3 | Done — trace v2, failure attribution |
| 4 | Done — eval compare, breakdown, exemplars |
| 5 | In progress — trajectory visualization |
| 6+ | Deferred — self-repair, roles, Docker |


## Upstream boundary

- Do **not** mix RepoPilot logic into `upstream/` unless fixing a deliberate upstream fork change.
- Baseline agent behavior = unmodified `mini` from `upstream/`.
- Upstream docs: `upstream/docs/`; upstream tests: `upstream/tests/`.

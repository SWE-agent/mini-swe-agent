# RepoPilot Benchmarks

Reproducible task definitions for baseline and RepoPilot evaluation.

## Layout

Each task is a directory:

```text
benchmarks/
  task_001_sudoku/
    config.yaml      # TaskConfig schema (validated by repopilot.schema)
    issue.md         # Prompt passed to mini / RepoPilot agent
    setup.patch      # Optional: applied after checkout to plant a known bug
```

Run outputs go to `runs/{task_id}/` (not committed).

## TaskConfig schema

See `src/repopilot/schema/task.py` for the Pydantic model.

| Field | Required | Description |
|-------|----------|-------------|
| `task_id` | yes | Must match directory name |
| `repo.path` or `repo.repo_url` | one required | Where the agent runs |
| `repo.base_commit` | yes | Git SHA to checkout before the run |
| `repo.setup_patch` | no | Patch applied after checkout (relative to task dir) |
| `issue_file` | yes | Task prompt file (default `issue.md`) |
| `test_command` | yes | Post-run verification command |
| `expected_behavior` | no | Human-readable success criteria |
| `agent.mode` | yes | `baseline` (upstream mini) or `repopilot` (later) |
| `agent.mini_flags` | no | Extra CLI args for `mini` |
| `agent.output_trajectory` | no | Trajectory path; `{task_id}` is substituted |

## Load a task in Python

```python
from pathlib import Path
from repopilot.schema import load_task

task = load_task(Path("benchmarks/task_001_sudoku"))
print(task.read_issue())
print(task.agent.resolve_output_trajectory(task.task_id))
```

## Tasks

| ID | Description | base_commit | Notes |
|----|-------------|-------------|-------|
| `task_001_sudoku` | Fix hint off-by-one in sudoku | `179e790` | Uses `setup.patch` to plant bug |
| `task_002_eval_module` | Fix multiplication bug in eval_expr | `61e468b` | Uses `setup.patch` to plant bug |
| `task_003_expr_multi` | Fix multiplication bug in expr package | `61e468b` | Multi-file package; same bug as task_002 |
| `task_004_expr_api_mismatch` | Fix tokenize/evaluate operator contract | `1e030a2` | `*` tokenized as `mul`; evaluator expects `*` |
| `task_005_serialize_unset` | Fix nested UNSET merge in serialize | `1e030a2` | `recursive_merge` skips nested UNSET filtering |

### task_001_sudoku

- **Repo:** this repository (`.`)
- **Bug:** `setup.patch` changes `hint()` to return `value + 1`
- **Verify:** `python -m pytest upstream/tests/run/test_sudoku.py -v`
- **Agent:** upstream `mini` in baseline mode (runner: Phase 1.5)

Run via CLI:

```bash
repopilot run task_001_sudoku
# or
repopilot-baseline run task_001_sudoku
```

Dry-run (print commands only):

```bash
repopilot run task_001_sudoku --dry-run
```

Skip agent (workspace prep + verify only; expect verify failure):

```bash
repopilot run task_001_sudoku --skip-mini
```

Re-record trace artifacts from an existing trajectory:

```bash
repopilot trace runs/task_001_sudoku/trajectory.traj.json
```

Phase 2 outputs (also written automatically after each run):

| File | Description |
|------|-------------|
| `trace.json` | Structured steps, tool calls, metrics |
| `patch.diff` | Extracted/reconstructed patch |
| `test.log` | Pytest runs from trajectory + runner verify |
| `final_report.md` | Human-readable run summary |

Pre-run check (manual):

```bash
git checkout 179e790
git apply benchmarks/task_001_sudoku/setup.patch
python -m pytest upstream/tests/run/test_sudoku.py -v   # expect 1 failed
git checkout -   # restore branch
```

### task_002_eval_module

- **Repo:** this repository (`.`)
- **Bug:** `setup.patch` changes `*` branch to use `+=` instead of `*=`
- **Verify:** `python -m pytest upstream/tests/run/test_eval_module.py -v`
- **Agent:** upstream `mini` in baseline mode

Pre-run check (manual):

```bash
git checkout 61e468b
git apply benchmarks/task_002_eval_module/setup.patch
PYTHONPATH=upstream/src python -m pytest upstream/tests/run/test_eval_module.py -v   # expect 2 failed
git checkout -   # restore branch
```

### task_003_expr_multi

- **Repo:** this repository (`.`)
- **Bug:** `setup.patch` changes `*` branch in `expr/evaluate.py` to use `+=` instead of `*=`
- **Verify:** `python -m pytest upstream/tests/run/test_expr.py -v`
- **Agent:** upstream `mini` in baseline mode; implementation split across `run/expr/` (`tokenize.py`, `evaluate.py`, …)

Pre-run check (manual):

```bash
git checkout 61e468b
git apply benchmarks/task_003_expr_multi/setup.patch
PYTHONPATH=upstream/src python -m pytest upstream/tests/run/test_expr.py -v   # expect 2 failed
git checkout -   # restore branch
```

### task_004_expr_api_mismatch

- **Repo:** this repository (`.`)
- **Bug:** `setup.patch` makes `tokenize.py` emit `mul` for `*` while `evaluate.py` expects `*`
- **Verify:** `python -m pytest upstream/tests/run/test_expr.py -v`
- **Agent:** must trace across `tokenize.py` and `evaluate.py`

Pre-run check (manual):

```bash
git checkout 1e030a2
git apply benchmarks/task_004_expr_api_mismatch/setup.patch
PYTHONPATH=upstream/src python -m pytest upstream/tests/run/test_expr.py -v   # expect 2 failed
git checkout -
```

### task_005_serialize_unset

- **Repo:** this repository (`.`)
- **Bug:** `setup.patch` assigns nested dicts without filtering nested `UNSET` values
- **Verify:** `python -m pytest upstream/tests/utils/test_serialize.py -v`
- **Agent:** single-file fix in `utils/serialize.py`

Pre-run check (manual):

```bash
git checkout 1e030a2
git apply benchmarks/task_005_serialize_unset/setup.patch
PYTHONPATH=upstream/src python -m pytest upstream/tests/utils/test_serialize.py -v   # expect 1 failed
git checkout -
```

## Adding a new task

1. Create `benchmarks/task_XXX_name/`
2. Add `config.yaml`, `issue.md`, and optional `setup.patch`
3. Ensure `task_id` in yaml matches the directory name
4. Add a row to the table above

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

| ID | Description | base_commit | failure_mode | difficulty |
|----|-------------|-------------|--------------|------------|
| `task_001_sudoku` | Fix hint off-by-one in sudoku | `179e790` | off_by_one | single_file |
| `task_002_eval_module` | Fix multiplication bug in eval_expr | `61e468b` | wrong_operator | single_file |
| `task_003_expr_multi` | Fix multiplication bug in expr package | `61e468b` | wrong_operator | multi_file |
| `task_004_expr_api_mismatch` | Fix tokenize/evaluate operator contract | `1e030a2` | api_mismatch | multi_file |
| `task_005_serialize_unset` | Fix nested UNSET merge in serialize | `1e030a2` | edge_case | single_file |
| `task_006_sudoku_logic` | Fix sudoku column validation logic | `179e790` | logic | single_file |
| `task_007_eval_import` | Fix eval_module facade import path | `61e468b` | import_path | multi_file |
| `task_008_expr_divzero` | Fix division-by-zero guard | `1e030a2` | wrong_condition | single_file |
| `task_009_serialize_none` | Fix recursive_merge None handling | `1e030a2` | null_handling | single_file |
| `task_010_expr_whitespace` | Fix expr whitespace normalization | `1e030a2` | logic | multi_file |

### Multi-bug tasks (3–5 bugs per task)

| ID | Description | base_commit | bug_count | difficulty |
|----|-------------|-------------|-----------|------------|
| `task_011_sudoku_multi3` | Fix three sudoku logic bugs | `179e790` | 3 | single_file |
| `task_012_expr_multi4` | Fix four expr package bugs | `1e030a2` | 4 | multi_file |
| `task_013_sudoku_multi5` | Fix five sudoku logic bugs | `179e790` | 5 | single_file |
| `task_014_expr_multi5` | Fix five expr package bugs | `1e030a2` | 5 | multi_file |

Multi-bug tasks use `eval.failure_mode: composite` and `eval.bug_count`. The issue prompt tells the agent to fix **all** injected bugs, not just the first failure.

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

### task_006_sudoku_logic

- **Repo:** this repository (`.`)
- **Bug:** `setup.patch` uses wrong index in column validation (`board[r][row]` instead of `board[r][col]`)
- **Verify:** `python -m pytest upstream/tests/run/test_sudoku.py -v`

Pre-run check (manual):

```bash
git checkout 179e790
git apply benchmarks/task_006_sudoku_logic/setup.patch
python -m pytest upstream/tests/run/test_sudoku.py -v   # expect failures
git checkout -
```

### task_007_eval_import

- **Repo:** this repository (`.`)
- **Bug:** `setup.patch` breaks `eval_module.py` facade import (`_expr_legacy` module does not exist)
- **Verify:** `python -m pytest upstream/tests/run/test_eval_module.py -v`

Pre-run check (manual):

```bash
git checkout 61e468b
git apply benchmarks/task_007_eval_import/setup.patch
PYTHONPATH=upstream/src python -m pytest upstream/tests/run/test_eval_module.py -v   # expect import/error failures
git checkout -
```

### task_008_expr_divzero

- **Repo:** this repository (`.`)
- **Bug:** `setup.patch` checks `result == 0` instead of `num.number == 0` before division
- **Verify:** `python -m pytest upstream/tests/run/test_expr.py -v`

Pre-run check (manual):

```bash
git checkout 1e030a2
git apply benchmarks/task_008_expr_divzero/setup.patch
PYTHONPATH=upstream/src python -m pytest upstream/tests/run/test_expr.py -v   # expect failures
git checkout -
```

### task_009_serialize_none

- **Repo:** this repository (`.`)
- **Bug:** `setup.patch` skips `None` values during merge (should allow explicit `None`)
- **Verify:** `python -m pytest upstream/tests/utils/test_serialize.py -v`

Pre-run check (manual):

```bash
git checkout 1e030a2
git apply benchmarks/task_009_serialize_none/setup.patch
PYTHONPATH=upstream/src python -m pytest upstream/tests/utils/test_serialize.py -v   # expect failures
git checkout -
```

### task_010_expr_whitespace

- **Repo:** this repository (`.`)
- **Bug:** `setup.patch` uses `strip()` instead of removing all spaces before tokenize
- **Verify:** `python -m pytest upstream/tests/run/test_expr.py -v`

Pre-run check (manual):

```bash
git checkout 1e030a2
git apply benchmarks/task_010_expr_whitespace/setup.patch
PYTHONPATH=upstream/src python -m pytest upstream/tests/run/test_expr.py -v   # expect failures
git checkout -
```

### task_011_sudoku_multi3

- **Repo:** this repository (`.`)
- **Bugs (3):** column index in `is_valid_move`, off-by-one `hint()`, incomplete `is_valid_group` range
- **Verify:** `python -m pytest upstream/tests/run/test_sudoku.py -v`

### task_012_expr_multi4

- **Repo:** this repository (`.`)
- **Bugs (4):** whitespace strip, div-zero guard, `*` token operator, `*=` in evaluate
- **Verify:** `python -m pytest upstream/tests/run/test_expr.py -v`

### task_013_sudoku_multi5

- **Repo:** this repository (`.`)
- **Bugs (5):** all three from task_011 plus `clear_value` ignores fixed cells, `set_value` skips validation
- **Verify:** `python -m pytest upstream/tests/run/test_sudoku.py -v`

### task_014_expr_multi5

- **Repo:** this repository (`.`)
- **Bugs (5):** all four from task_012 plus `-=` mapped to `+=` in evaluate
- **Verify:** `python -m pytest upstream/tests/run/test_expr.py -v`

## Adding a new task

1. Create `benchmarks/task_XXX_name/`
2. Add `config.yaml`, `issue.md`, and optional `setup.patch`
3. Ensure `task_id` in yaml matches the directory name
4. Add a row to the table above

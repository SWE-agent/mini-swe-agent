#!/usr/bin/env python3
"""Verify benchmark setup patches apply and fail tests as expected."""

from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

CASES = [
    {
        "name": "task_006",
        "base": "179e790",
        "patch": ROOT / "benchmarks/task_006_sudoku_logic/setup.patch",
        "test": "python -m pytest upstream/tests/run/test_sudoku.py -q",
    },
    {
        "name": "task_007",
        "base": "61e468b",
        "patch": ROOT / "benchmarks/task_007_eval_import/setup.patch",
        "test": "python -m pytest upstream/tests/run/test_eval_module.py -q",
    },
    {
        "name": "task_008",
        "base": "1e030a2",
        "patch": ROOT / "benchmarks/task_008_expr_divzero/setup.patch",
        "test": "python -m pytest upstream/tests/run/test_expr.py -q",
    },
    {
        "name": "task_009",
        "base": "1e030a2",
        "patch": ROOT / "benchmarks/task_009_serialize_none/setup.patch",
        "test": "python -m pytest upstream/tests/utils/test_serialize.py -q",
    },
    {
        "name": "task_010",
        "base": "1e030a2",
        "patch": ROOT / "benchmarks/task_010_expr_whitespace/setup.patch",
        "test": "python -m pytest upstream/tests/run/test_expr.py -q",
    },
    {
        "name": "task_011",
        "base": "179e790",
        "patch": ROOT / "benchmarks/task_011_sudoku_multi3/setup.patch",
        "test": "python -m pytest upstream/tests/run/test_sudoku.py -q",
    },
    {
        "name": "task_012",
        "base": "1e030a2",
        "patch": ROOT / "benchmarks/task_012_expr_multi4/setup.patch",
        "test": "python -m pytest upstream/tests/run/test_expr.py -q",
    },
    {
        "name": "task_013",
        "base": "179e790",
        "patch": ROOT / "benchmarks/task_013_sudoku_multi5/setup.patch",
        "test": "python -m pytest upstream/tests/run/test_sudoku.py -q",
    },
    {
        "name": "task_014",
        "base": "1e030a2",
        "patch": ROOT / "benchmarks/task_014_expr_multi5/setup.patch",
        "test": "python -m pytest upstream/tests/run/test_expr.py -q",
    },
]


def verify(case: dict) -> bool:
    patch_path = case["patch"]
    if not patch_path.is_file():
        print(f"SKIP {case['name']}: missing {patch_path}")
        return False
    with tempfile.TemporaryDirectory() as td:
        wt = Path(td) / "ws"
        subprocess.run(
            ["git", "worktree", "add", "--detach", str(wt), case["base"]],
            cwd=ROOT,
            check=True,
            capture_output=True,
        )
        try:
            r = subprocess.run(["git", "apply", str(patch_path)], cwd=wt, capture_output=True, text=True)
            if r.returncode != 0:
                print(f"FAIL {case['name']}: patch apply\n{r.stderr}")
                return False
            env = {**subprocess.os.environ, "PYTHONPATH": str(wt / "upstream/src")}
            r = subprocess.run(case["test"], shell=True, cwd=wt, env=env, capture_output=True, text=True)
            ok = r.returncode != 0
            print(f"{'OK' if ok else 'FAIL'} {case['name']}: pytest rc={r.returncode}")
            if not ok:
                print(r.stdout[-500:])
            return ok
        finally:
            subprocess.run(["git", "worktree", "remove", "--force", str(wt)], cwd=ROOT, capture_output=True)


if __name__ == "__main__":
    results = [verify(c) for c in CASES]
    raise SystemExit(0 if all(results) else 1)

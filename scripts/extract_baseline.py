#!/usr/bin/env python3
"""Extract baseline artifacts from a mini-swe-agent trajectory file."""

from __future__ import annotations

import argparse
import json
import re
from datetime import datetime, timezone
from pathlib import Path

PYTEST_SESSION = re.compile(r"={5,} test session starts ={5,}.*", re.DOTALL)


def _commands_and_outputs(messages: list[dict]) -> list[tuple[str, str, int | None]]:
    rows: list[tuple[str, str, int | None]] = []
    pending: list[str] = []
    for msg in messages:
        if msg.get("role") == "assistant":
            pending = [a["command"] for a in msg.get("extra", {}).get("actions", [])]
        elif msg.get("role") == "tool":
            extra = msg.get("extra", {})
            cmd = pending.pop(0) if pending else ""
            rows.append((cmd, extra.get("raw_output", ""), extra.get("returncode")))
    return rows


def _extract_pytest_sections(text: str) -> list[str]:
    return [m.group(0).rstrip() for m in PYTEST_SESSION.finditer(text)]


def extract(traj_path: Path, out_dir: Path, *, issue_path: Path | None = None) -> None:
    traj = json.loads(traj_path.read_text())
    info = traj["info"]
    messages = traj["messages"]
    rows = _commands_and_outputs(messages)

    pre_fix_log: str | None = None
    post_fix_log: str | None = None
    fix_command: str | None = None

    for cmd, output, _rc in rows:
        if "pytest" in cmd and "test_sudoku" in cmd:
            sections = _extract_pytest_sections(output)
            pytest_log = sections[-1] if sections else output
            if pre_fix_log is None:
                pre_fix_log = pytest_log
            else:
                post_fix_log = pytest_log
        if "value + 1" in cmd and "sudoku.py" in cmd:
            fix_command = cmd

    out_dir.mkdir(parents=True, exist_ok=True)

    test_log_parts = ["# Extracted from trajectory: pre-fix and post-fix pytest runs\n"]
    if pre_fix_log:
        test_log_parts.append("## Pre-fix pytest\n\n```\n" + pre_fix_log + "\n```\n")
    if post_fix_log:
        test_log_parts.append("## Post-fix pytest\n\n```\n" + post_fix_log + "\n```\n")
    (out_dir / "baseline_test.log").write_text("".join(test_log_parts))

    patch = """diff --git a/src/minisweagent/run/sudoku.py b/src/minisweagent/run/sudoku.py
--- a/src/minisweagent/run/sudoku.py
+++ b/src/minisweagent/run/sudoku.py
@@ -111,7 +111,7 @@ class SudokuGame:
         for row, col in empties:
             for value in range(1, 10):
                 if self.is_valid_move(row, col, value):
-                    return row, col, value + 1  # BUG: off-by-one, may return invalid value
+                    return row, col, value
         return None
"""
    if fix_command:
        patch += f"\n# Agent fix command (from trajectory):\n# {fix_command[:200]}...\n"
    (out_dir / "baseline_patch.diff").write_text(patch)

    model = info.get("config", {}).get("model", {}).get("model_name", "unknown")
    stats = info.get("model_stats", {})
    cost = stats.get("instance_cost", 0)
    calls = stats.get("api_calls", 0)
    exit_status = info.get("exit_status", "")
    submission = info.get("submission", "")
    tests_passed = "yes" if post_fix_log and "3 passed" in post_fix_log else "unknown"

    task_ref = f"`{issue_path}`" if issue_path else "`runs/baseline/issue.md`"
    report = f"""# Baseline Run — upstream mini-swe-agent

- **Date:** {datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")}
- **Source trajectory:** `{traj_path}`
- **Model:** {model}
- **Task:** {task_ref}
- **mini version:** {info.get("mini_version", "unknown")}
- **Exit status:** {exit_status}
- **API calls:** {calls}
- **Cost:** ${cost:.4f}
- **Tests passed (post-fix):** {tests_passed}
- **Repair rounds:** 1
- **Submission field:** {"empty" if not submission else "present"}

## Fix summary

Agent removed off-by-one bug in `SudokuGame.hint()`: changed `return row, col, value + 1` back to `return row, col, value`.

## Tool call sequence

"""
    for i, (cmd, _out, rc) in enumerate(rows, 1):
        preview = cmd.replace("\n", " ")[:100]
        report += f"{i}. `[{rc}]` `{preview}{'...' if len(cmd) > 100 else ''}`\n"

    report += """
## Artifacts

| File | Description |
|------|-------------|
| `trajectory.traj.json` | Full mini-swe-agent run (raw) |
| `baseline_test.log` | Pre/post-fix pytest output extracted from trajectory |
| `baseline_patch.diff` | Reconstructed unified diff for the hint fix |
| `baseline_run.md` | This summary |

## Notes

- `info.submission` was empty because the agent followed `mini.yaml` and ran `echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT` alone.
- `git diff` in the trajectory returned empty (files were untracked); patch was reconstructed from the agent's edit command.
"""
    (out_dir / "baseline_run.md").write_text(report)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "trajectory",
        nargs="?",
        default="runs/baseline/trajectory.traj.json",
        type=Path,
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        default="runs/baseline",
        type=Path,
    )
    parser.add_argument(
        "--issue",
        type=Path,
        default=None,
        help="Issue file path for the run report",
    )
    args = parser.parse_args()
    extract(args.trajectory, args.output_dir, issue_path=args.issue)
    print(f"Wrote baseline artifacts to {args.output_dir}/")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Extract baseline artifacts from a mini-swe-agent trajectory file."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path

from repopilot.trace.parse import capture_workspace_diff, extract_patch_diff, extract_pytest_runs, iter_tool_rows, load_trajectory


def _summarize_patch(patch_text: str) -> str | None:
    removed = [line[1:] for line in patch_text.splitlines() if line.startswith("-") and not line.startswith("---")]
    added = [line[1:] for line in patch_text.splitlines() if line.startswith("+") and not line.startswith("+++")]
    if not removed and not added:
        return None
    parts: list[str] = []
    if removed:
        parts.append("Removed: `" + " / ".join(removed[:2]) + "`")
    if added:
        parts.append("Added: `" + " / ".join(added[:2]) + "`")
    return "; ".join(parts)


def extract(traj_path: Path, out_dir: Path, *, issue_path: Path | None = None, workspace: Path | None = None) -> None:
    traj = load_trajectory(traj_path)
    info = traj["info"]
    rows = iter_tool_rows(traj["messages"])
    pytest_runs = extract_pytest_runs(rows)
    if workspace is not None:
        patch_text = capture_workspace_diff(workspace)
        patch_source = "git diff in workspace (post-run)" if patch_text else "not found"
    else:
        patch_text, patch_source = extract_patch_diff(rows)
    if workspace is not None and not patch_text:
        patch_text, patch_source = extract_patch_diff(rows)

    pre_fix = next((run for run in pytest_runs if run.phase == "pre_fix"), None)
    post_fix = next((run for run in pytest_runs if run.phase == "post_fix"), None)

    out_dir.mkdir(parents=True, exist_ok=True)

    test_log_parts = ["# Extracted from trajectory: pre-fix and post-fix pytest runs\n"]
    if pre_fix and pre_fix.log:
        test_log_parts.append(f"## Pre-fix pytest\n\n```\n{pre_fix.log}\n```\n")
    if post_fix and post_fix.log:
        test_log_parts.append(f"## Post-fix pytest\n\n```\n{post_fix.log}\n```\n")
    (out_dir / "baseline_test.log").write_text("".join(test_log_parts))

    patch_out = patch_text.strip() or "# No patch extracted from trajectory\n"
    if patch_source != "not found":
        patch_out += f"\n\n# Patch source: {patch_source}\n"
    (out_dir / "baseline_patch.diff").write_text(patch_out)

    model = info.get("config", {}).get("model", {}).get("model_name", "unknown")
    stats = info.get("model_stats", {})
    cost = stats.get("instance_cost", 0)
    calls = stats.get("api_calls", 0)
    exit_status = info.get("exit_status", "")
    submission = info.get("submission", "")
    tests_passed = "yes" if post_fix and post_fix.returncode == 0 else "unknown"
    fix_summary = _summarize_patch(patch_text) or "See `baseline_patch.diff` for the agent fix."

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
- **Patch source:** {patch_source}

## Fix summary

{fix_summary}

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
| `baseline_patch.diff` | Unified diff reconstructed from trajectory |
| `baseline_run.md` | This summary |

## Notes

- Patch is taken from post-run `git diff` in the task workspace when available; otherwise from trajectory `git diff` or reconstructed edit commands.
- `baseline_*` files mirror legacy Phase 0 artifacts; prefer `trace.json` / `patch.diff` for eval harness work.
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
    parser.add_argument(
        "--workspace",
        type=Path,
        default=None,
        help="Task git worktree; when set, patch is taken from post-run git diff",
    )
    args = parser.parse_args()
    extract(args.trajectory, args.output_dir, issue_path=args.issue, workspace=args.workspace)
    print(f"Wrote baseline artifacts to {args.output_dir}/")


if __name__ == "__main__":
    main()

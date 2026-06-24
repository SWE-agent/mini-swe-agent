"""Record structured trace artifacts from agent trajectories (Phase 2–3)."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

from repopilot.trace.classify import build_failure_reason_md, classify_trace
from repopilot.trace.parse import (
    PytestRun,
    TraceStep,
    extract_patch_diff,
    extract_pytest_runs,
    extract_viewed_files,
    infer_step_stage,
    extract_files_touched,
    iter_tool_rows,
    iter_trace_steps,
    load_trajectory,
)

SCHEMA_VERSION = "2.0"


@dataclass
class TraceContext:
    task_id: str | None = None
    issue_path: str | None = None
    verify_test_log: Path | None = None
    tests_passed: bool | None = None
    agent_mode: str = "baseline"
    failure_mode: str | None = None
    difficulty: str | None = None
    eval_tags: list[str] | None = None


@dataclass
class TraceArtifacts:
    trace_json: Path
    patch_diff: Path
    test_log: Path
    final_report: Path
    failure_reason: Path


def _count_repair_rounds(pytest_runs: list[PytestRun], rows: list[tuple[str, str, int | None]]) -> int:
    failed_runs = sum(1 for run in pytest_runs if run.returncode not in (None, 0))
    if failed_runs == 0:
        return 0
    edit_commands = sum(
        1
        for cmd, _out, _rc in rows
        if any(token in cmd for token in ("path.write_text", "sed -i", "git apply", "patch"))
    )
    return max(edit_commands, 1 if failed_runs else 0)


def _build_test_log(pytest_runs: list[PytestRun], verify_log: Path | None) -> str:
    parts = ["# Test log extracted from agent trajectory\n"]
    for run in pytest_runs:
        parts.append(f"## {run.phase} (exit {run.returncode})\n\n```\n{run.log}\n```\n\n")
        if run.failed_tests:
            parts.append("Failed tests:\n")
            for failure in run.failed_tests:
                line = f"- `{failure.test}`"
                if failure.assertion:
                    line += f": {failure.assertion}"
                parts.append(line + "\n")
            parts.append("\n")
    if verify_log and verify_log.is_file():
        parts.append("## runner_verify\n\n```\n")
        parts.append(verify_log.read_text())
        parts.append("\n```\n")
    return "".join(parts)


def _steps_to_dict(steps: list[TraceStep]) -> list[dict]:
    result: list[dict] = []
    for step in steps:
        tool_calls = [asdict(tc) for tc in step.tool_calls]
        files_touched: list[str] = []
        seen: set[str] = set()
        for tc in step.tool_calls:
            for path in extract_files_touched(tc.command, tc.output):
                if path not in seen:
                    seen.add(path)
                    files_touched.append(path)
        result.append(
            {
                "step": step.step,
                "stage": infer_step_stage(step.tool_calls),
                "reasoning": step.reasoning,
                "step_cost": step.step_cost,
                "files_touched": files_touched,
                "tool_calls": tool_calls,
            }
        )
    return result


def _pytest_to_dict(runs: list[PytestRun]) -> list[dict]:
    return [
        {
            "phase": run.phase,
            "returncode": run.returncode,
            "summary": run.summary,
            "log": run.log,
            "failed_tests": [asdict(f) for f in run.failed_tests],
        }
        for run in runs
    ]


def build_trace_document(
    traj: dict,
    *,
    ctx: TraceContext,
    traj_path: Path,
) -> dict:
    messages = traj["messages"]
    info = traj["info"]
    rows = iter_tool_rows(messages)
    steps = iter_trace_steps(messages)
    pytest_runs = extract_pytest_runs(rows)
    patch_text, patch_source = extract_patch_diff(rows)
    stats = info.get("model_stats", {})
    config = info.get("config", {})

    steps_dict = _steps_to_dict(steps)
    pytest_dict = _pytest_to_dict(pytest_runs)
    patch = {"source": patch_source, "text": patch_text}

    task_tags: dict[str, object] = {}
    if ctx.failure_mode:
        task_tags["failure_mode"] = ctx.failure_mode
    if ctx.difficulty:
        task_tags["difficulty"] = ctx.difficulty
    if ctx.eval_tags:
        task_tags["tags"] = list(ctx.eval_tags)

    outcome_fields = classify_trace(
        steps=steps_dict,
        pytest_runs=pytest_dict,
        patch=patch,
        exit_status=str(info.get("exit_status") or ""),
        tests_passed=ctx.tests_passed,
    )

    trace: dict = {
        "schema_version": SCHEMA_VERSION,
        "recorded_at": datetime.now(timezone.utc).isoformat(),
        "task_id": ctx.task_id,
        "issue_path": ctx.issue_path,
        "agent_mode": ctx.agent_mode,
        "source_trajectory": str(traj_path),
        "model": config.get("model", {}).get("model_name", "unknown"),
        "mini_version": info.get("mini_version", "unknown"),
        "exit_status": info.get("exit_status", ""),
        "submission": info.get("submission", ""),
        "task_tags": task_tags,
        **outcome_fields,
        "metrics": {
            "api_calls": stats.get("api_calls", 0),
            "instance_cost": stats.get("instance_cost", 0.0),
            "repair_rounds": _count_repair_rounds(pytest_runs, rows),
            "step_count": len(steps),
            "tests_passed": ctx.tests_passed,
        },
        "retrieved_files": extract_viewed_files(rows),
        "steps": steps_dict,
        "pytest_runs": pytest_dict,
        "patch": patch,
    }
    return trace


def build_final_report(trace: dict) -> str:
    metrics = trace["metrics"]
    pytest_runs = trace["pytest_runs"]
    post_fix = next((run for run in pytest_runs if run["phase"] == "post_fix"), None)
    pre_fix = next((run for run in pytest_runs if run["phase"] == "pre_fix"), None)

    lines = [
        "# RepoPilot Run Report",
        "",
        f"- **Task:** `{trace.get('task_id', 'unknown')}`",
        f"- **Issue:** `{trace.get('issue_path', 'unknown')}`",
        f"- **Recorded:** {trace['recorded_at']}",
        f"- **Model:** {trace['model']}",
        f"- **Agent mode:** {trace['agent_mode']}",
        f"- **Outcome:** {trace.get('outcome', 'unknown')}",
        f"- **mini version:** {trace['mini_version']}",
        f"- **Exit status:** {trace['exit_status']}",
        f"- **API calls:** {metrics['api_calls']}",
        f"- **Cost:** ${metrics['instance_cost']:.4f}",
        f"- **Repair rounds:** {metrics['repair_rounds']}",
        f"- **Tests passed (agent):** {'yes' if post_fix and post_fix.get('returncode') == 0 else 'no'}",
        f"- **Tests passed (runner verify):** {'yes' if metrics.get('tests_passed') else 'no' if metrics.get('tests_passed') is False else 'unknown'}",
        "",
    ]

    if trace.get("failure_category"):
        lines.append(f"- **Failure category:** `{trace['failure_category']}`")
    if trace.get("failure_stage"):
        lines.append(f"- **Failure stage:** `{trace['failure_stage']}`")
    if trace.get("failed_step") is not None:
        lines.append(f"- **Failed step:** {trace['failed_step']}")
    if trace.get("failure_message"):
        lines.append(f"- **Failure message:** {trace['failure_message']}")
    if trace.get("task_tags"):
        lines.append(f"- **Task tags:** `{trace['task_tags']}`")
    lines.append("")

    if pre_fix:
        lines.extend([f"- **Pre-fix tests:** {pre_fix.get('summary') or 'see test.log'}", ""])
    if post_fix:
        lines.extend([f"- **Post-fix tests:** {post_fix.get('summary') or 'see test.log'}", ""])

    lines.extend(["## Agent steps", ""])
    for step in trace["steps"]:
        stage = step.get("stage", "other")
        lines.append(f"### Step {step['step']} ({stage})")
        if step["reasoning"]:
            lines.append(step["reasoning"])
            lines.append("")
        if step.get("files_touched"):
            lines.append("Files: " + ", ".join(f"`{p}`" for p in step["files_touched"]))
            lines.append("")
        for i, tc in enumerate(step["tool_calls"], 1):
            preview = tc["command"].replace("\n", " ")[:120]
            lines.append(f"{i}. `[{tc['returncode']}]` `{preview}{'…' if len(tc['command']) > 120 else ''}`")
        lines.append("")

    if trace["retrieved_files"]:
        lines.extend(["## Retrieved files", ""])
        for path in trace["retrieved_files"]:
            lines.append(f"- `{path}`")
        lines.append("")

    lines.extend(
        [
            "## Artifacts",
            "",
            "| File | Description |",
            "|------|-------------|",
            "| `trace.json` | Structured step/tool/test/patch trace (schema v2) |",
            "| `failure_reason.md` | Pytest-grounded failure explanation |",
            "| `patch.diff` | Best-effort unified diff |",
            "| `test.log` | Pytest output from trajectory (+ runner verify) |",
            "| `final_report.md` | This report |",
            "| `trajectory.traj.json` | Raw mini-swe-agent output |",
            "",
            f"Patch source: {trace['patch']['source']}",
        ]
    )
    return "\n".join(lines) + "\n"


def record_trace(
    trajectory_path: Path,
    output_dir: Path,
    *,
    ctx: TraceContext | None = None,
) -> TraceArtifacts:
    ctx = ctx or TraceContext()
    trajectory_path = trajectory_path.resolve()
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    traj = load_trajectory(trajectory_path)
    rows = iter_tool_rows(traj["messages"])
    pytest_runs = extract_pytest_runs(rows)
    trace = build_trace_document(traj, ctx=ctx, traj_path=trajectory_path)

    trace_json = output_dir / "trace.json"
    patch_diff = output_dir / "patch.diff"
    test_log = output_dir / "test.log"
    final_report = output_dir / "final_report.md"
    failure_reason = output_dir / "failure_reason.md"

    trace_json.write_text(json.dumps(trace, indent=2, ensure_ascii=False) + "\n")
    patch_diff.write_text(trace["patch"]["text"] or "# No patch extracted from trajectory\n")
    test_log.write_text(_build_test_log(pytest_runs, ctx.verify_test_log))
    final_report.write_text(build_final_report(trace))
    failure_reason.write_text(build_failure_reason_md(trace))

    return TraceArtifacts(
        trace_json=trace_json,
        patch_diff=patch_diff,
        test_log=test_log,
        final_report=final_report,
        failure_reason=failure_reason,
    )

"""Trajectory metrics from trace v2 steps (Phase 4)."""

from __future__ import annotations

from repopilot.eval.loader import RunRecord
from repopilot.eval.visualize import render_mermaid_timeline
from repopilot.trace.parse import is_edit_command


def _step_has_edit(step: dict) -> bool:
    return any(is_edit_command(tc.get("command", "")) for tc in step.get("tool_calls", []))


def _step_has_pytest(step: dict) -> bool:
    return any("pytest" in tc.get("command", "").lower() for tc in step.get("tool_calls", []))


def trajectory_metrics(record: RunRecord) -> dict:
    """Compute steps-to-first-edit, files touched, and related stats."""
    steps = record.steps
    first_edit_step: int | None = None
    first_test_step: int | None = None
    tests_before_edit = 0
    files_seen: list[str] = []
    seen: set[str] = set()

    for step in steps:
        step_num = step.get("step")
        if first_test_step is None and _step_has_pytest(step):
            first_test_step = step_num
        if first_edit_step is None and _step_has_edit(step):
            first_edit_step = step_num
        if first_edit_step is None and _step_has_pytest(step):
            tests_before_edit += 1
        for path in step.get("files_touched") or []:
            if path not in seen:
                seen.add(path)
                files_seen.append(path)

    for path in record.retrieved_files:
        if path not in seen:
            seen.add(path)
            files_seen.append(path)

    all_files = files_seen
    stages = [step.get("stage", "other") for step in steps if step.get("stage")]

    return {
        "step_count": record.step_count,
        "first_test_step": first_test_step,
        "steps_to_first_edit": first_edit_step,
        "tests_before_edit": tests_before_edit,
        "files_touched_count": len(all_files),
        "files_touched": all_files,
        "stages": stages,
        "repair_rounds": record.repair_rounds,
    }


def render_trajectory_analysis(record: RunRecord) -> str:
    metrics = trajectory_metrics(record)
    lines = [
        f"# Trajectory Analysis — {record.task_id}",
        "",
        f"- **Agent mode:** {record.agent_mode}",
        f"- **Outcome:** {record.outcome}",
        f"- **Steps:** {metrics['step_count']}",
        f"- **First test step:** {metrics['first_test_step'] or '—'}",
        f"- **Steps to first edit:** {metrics['steps_to_first_edit'] or '—'}",
        f"- **Pytest runs before first edit:** {metrics['tests_before_edit']}",
        f"- **Files touched (unique):** {metrics['files_touched_count']}",
        f"- **Repair rounds:** {metrics['repair_rounds']}",
    ]
    if record.failure_mode:
        lines.append(f"- **Failure mode (tag):** `{record.failure_mode}`")
    if record.difficulty:
        lines.append(f"- **Difficulty (tag):** `{record.difficulty}`")
    lines.append("")

    if record.steps:
        lines.extend(["## Trajectory timeline (Mermaid)", "", render_mermaid_timeline(record), ""])

    if metrics["stages"]:
        lines.extend(["## Step stages", "", " → ".join(f"`{s}`" for s in metrics["stages"]), ""])

    if metrics["files_touched"]:
        lines.extend(["", "## Files touched", ""])
        for path in metrics["files_touched"]:
            lines.append(f"- `{path}`")
        lines.append("")

    if record.steps:
        lines.extend(["## Step timeline", ""])
        for step in record.steps:
            stage = step.get("stage", "other")
            lines.append(f"### Step {step.get('step')} ({stage})")
            if step.get("reasoning"):
                text = step["reasoning"]
                lines.append(text[:400] + ("…" if len(text) > 400 else ""))
                lines.append("")
            touched = step.get("files_touched") or []
            if touched:
                lines.append("Files: " + ", ".join(f"`{p}`" for p in touched))
                lines.append("")
        lines.append("")

    return "\n".join(lines)

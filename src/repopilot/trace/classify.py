"""Classify agent run outcomes from structured trace data (Phase 3)."""

from __future__ import annotations

from typing import Any


def _has_edit_command(steps: list[dict]) -> bool:
    for step in steps:
        for tc in step.get("tool_calls", []):
            cmd = tc.get("command", "")
            if any(token in cmd for token in ("path.write_text", "sed -i", "git apply", "patch")):
                return True
    return False


def _agent_post_fix_passed(pytest_runs: list[dict]) -> bool | None:
    post = next((r for r in pytest_runs if r.get("phase") == "post_fix"), None)
    if post is None:
        return None
    return post.get("returncode") == 0


def _last_pytest_summary(pytest_runs: list[dict]) -> str:
    if not pytest_runs:
        return ""
    last = pytest_runs[-1]
    summary = last.get("summary") or ""
    if summary:
        return summary
    log = last.get("log", "")
    for line in reversed(log.splitlines()):
        stripped = line.strip().strip("=")
        if "passed" in stripped or "failed" in stripped:
            return stripped
    return ""


def _find_failed_step(steps: list[dict], *, tests_passed: bool | None) -> int | None:
    if not steps:
        return None
    if tests_passed is False:
        for step in reversed(steps):
            for tc in step.get("tool_calls", []):
                cmd = tc.get("command", "")
                if "pytest" in cmd and tc.get("returncode") not in (None, 0):
                    return step.get("step")
        for step in reversed(steps):
            for tc in step.get("tool_calls", []):
                if tc.get("returncode") not in (None, 0):
                    return step.get("step")
    return steps[-1].get("step")


def classify_trace(
    *,
    steps: list[dict],
    pytest_runs: list[dict],
    patch: dict[str, Any],
    exit_status: str,
    tests_passed: bool | None,
) -> dict[str, Any]:
    """Return outcome, failure_category, failure_stage, failed_step, failure_message."""
    agent_post_fix = _agent_post_fix_passed(pytest_runs)
    has_pytest = bool(pytest_runs)
    has_edit = _has_edit_command(steps)
    patch_text = (patch.get("text") or "").strip()
    has_patch = bool(patch_text and not patch_text.startswith("#"))

    if tests_passed is True:
        return {
            "outcome": "success",
            "failure_category": None,
            "failure_stage": None,
            "failed_step": None,
            "failure_message": None,
        }

    result: dict[str, Any] = {"outcome": "failure"}

    if agent_post_fix is True and tests_passed is False:
        result.update(
            {
                "failure_category": "verify_mismatch",
                "failure_stage": "test",
                "failure_message": "Agent post-fix pytest passed; runner verify failed",
            }
        )
    elif not has_pytest:
        result.update(
            {
                "failure_category": "tests_never_run",
                "failure_stage": "read" if has_edit else "plan",
                "failure_message": "No pytest run found in trajectory",
            }
        )
    elif agent_post_fix is False or (
        pytest_runs and pytest_runs[-1].get("returncode") not in (None, 0)
    ):
        summary = _last_pytest_summary(pytest_runs)
        failed_tests = _collect_failed_test_names(pytest_runs)
        message = summary or "Tests still failing"
        if failed_tests:
            message = f"{summary or 'Tests still failing'} — failed: {', '.join(failed_tests[:3])}"
        result.update(
            {
                "failure_category": "tests_still_failing",
                "failure_stage": "test",
                "failure_message": message,
            }
        )
    elif not has_patch and not has_edit:
        result.update(
            {
                "failure_category": "patch_empty",
                "failure_stage": "edit",
                "failure_message": "No patch or edit command in trajectory",
            }
        )
    elif "cost" in exit_status.lower() or "limit" in exit_status.lower():
        result.update(
            {
                "failure_category": "cost_limit_exceeded",
                "failure_stage": "submit",
                "failure_message": exit_status,
            }
        )
    elif not has_edit and exit_status == "Submitted":
        result.update(
            {
                "failure_category": "agent_exited_early",
                "failure_stage": "edit",
                "failure_message": "Submitted without editing source",
            }
        )
    else:
        result.update(
            {
                "failure_category": "unknown",
                "failure_stage": "submit",
                "failure_message": exit_status or "Unknown failure",
            }
        )

    result["failed_step"] = _find_failed_step(steps, tests_passed=tests_passed)
    return result


def _collect_failed_test_names(pytest_runs: list[dict]) -> list[str]:
    names: list[str] = []
    seen: set[str] = set()
    for run in pytest_runs:
        for failure in run.get("failed_tests") or []:
            name = failure.get("test") if isinstance(failure, dict) else str(failure)
            if name and name not in seen:
                seen.add(name)
                names.append(name)
    return names


def build_failure_reason_md(trace: dict) -> str:
    """Render pytest-grounded failure explanation for a run."""
    lines = [
        "# Failure Reason",
        "",
        f"- **Task:** `{trace.get('task_id', 'unknown')}`",
        f"- **Outcome:** {trace.get('outcome', 'unknown')}",
    ]
    if trace.get("failure_category"):
        lines.append(f"- **Category:** `{trace['failure_category']}`")
    if trace.get("failure_stage"):
        lines.append(f"- **Stage:** `{trace['failure_stage']}`")
    if trace.get("failed_step") is not None:
        lines.append(f"- **Failed step:** {trace['failed_step']}")
    lines.append("")

    message = trace.get("failure_message")
    if message:
        lines.extend(["## Summary", "", message, ""])

    if trace.get("outcome") == "success":
        lines.extend(["All runner verification checks passed.", ""])
        return "\n".join(lines)

    pytest_runs = trace.get("pytest_runs") or []
    failures = [
        (run.get("phase"), failure)
        for run in pytest_runs
        for failure in (run.get("failed_tests") or [])
    ]
    if failures:
        lines.extend(["## Pytest failures (from trajectory)", ""])
        for phase, failure in failures:
            test = failure.get("test", "unknown")
            lines.append(f"### {test} ({phase})")
            if failure.get("assertion"):
                lines.append(f"- **Assertion:** `{failure['assertion']}`")
            if failure.get("file_line"):
                lines.append(f"- **Location:** `{failure['file_line']}`")
            lines.append("")

    verify_passed = trace.get("metrics", {}).get("tests_passed")
    agent_post = _agent_post_fix_passed(pytest_runs)
    if agent_post is True and verify_passed is False:
        lines.extend(
            [
                "## Verify mismatch",
                "",
                "Agent's last in-trajectory pytest passed, but runner verify failed.",
                "Check `verify_test.log` for environment or PYTHONPATH differences.",
                "",
            ]
        )

    failed_step = trace.get("failed_step")
    if failed_step is not None:
        step = next((s for s in trace.get("steps", []) if s.get("step") == failed_step), None)
        if step:
            lines.extend([f"## Step {failed_step} timeline", ""])
            if step.get("stage"):
                lines.append(f"- **Stage:** `{step['stage']}`")
            if step.get("files_touched"):
                lines.append("- **Files touched:**")
                for path in step["files_touched"]:
                    lines.append(f"  - `{path}`")
            if step.get("reasoning"):
                lines.append(f"- **Reasoning:** {step['reasoning'][:300]}")
            lines.append("")

    return "\n".join(lines) + "\n"

"""Failure taxonomy breakdown by task tags (Phase 4)."""

from __future__ import annotations

from collections import Counter, defaultdict

from repopilot.eval.loader import RunRecord
from repopilot.eval.metrics import failure_breakdown
from repopilot.trace.classify import _collect_edited_files, _collect_failing_source_files
from repopilot.trace.parse import extract_pytest_log, parse_pytest_failures

EXEMPLAR_CATEGORIES = ("tests_still_failing", "wrong_file_edited")

BY_FIELDS = {
    "failure_mode": "by_failure_mode",
    "difficulty": "by_difficulty",
    "failure_category": "by_category",
    "failure_stage": "by_stage",
}


def _read_test_log(record: RunRecord) -> str:
    log_path = record.run_dir / "test.log"
    if log_path.is_file():
        return log_path.read_text()
    parts: list[str] = []
    for run in record.pytest_runs:
        phase = run.get("phase", "unknown")
        log = run.get("log") or ""
        if log.strip():
            parts.append(f"=== {phase} ===\n{log.rstrip()}\n")
    return "\n".join(parts)


def extract_failure_snippet(record: RunRecord, *, max_lines: int = 20) -> str:
    """Pull a pytest-grounded snippet from test.log or trace pytest_runs."""
    raw = _read_test_log(record)
    if not raw.strip():
        return ""

    log = extract_pytest_log(raw) if "test session starts" in raw.lower() else raw
    lines = log.splitlines()
    if record.failure_category == "wrong_file_edited":
        edited = _collect_edited_files(record.steps)
        failing = _collect_failing_source_files(record.pytest_runs)
        header = [
            f"Edited: {', '.join(edited[:3]) or 'unknown'}",
            f"Failing test source: {', '.join(failing[:3]) or 'unknown'}",
            "",
        ]
        snippet = _failure_block_snippet(lines, max_lines=max_lines)
        return "\n".join(header + (snippet.splitlines() if snippet else []))

    snippet = _failure_block_snippet(lines, max_lines=max_lines)
    if snippet:
        return snippet

    failures = parse_pytest_failures(log)
    if failures:
        first = failures[0]
        parts = [f"FAILED {first.test}"]
        if first.assertion:
            parts.append(f"> {first.assertion}")
        if first.file_line:
            parts.append(first.file_line)
        return "\n".join(parts)
    return ""


def _failure_block_snippet(lines: list[str], *, max_lines: int) -> str:
    start = next((i for i, line in enumerate(lines) if "FAILURES" in line), None)
    if start is not None:
        block = lines[start : start + max_lines + 1]
        return "\n".join(block).strip()
    failed_idx = next((i for i, line in enumerate(lines) if " FAILED" in line), None)
    if failed_idx is not None:
        block = lines[max(0, failed_idx - 2) : failed_idx + max_lines]
        return "\n".join(block).strip()
    return ""


def build_category_exemplars(records: list[RunRecord]) -> dict[str, list[dict]]:
    """Collect exemplar runs for categories that benefit from test.log snippets."""
    exemplars: dict[str, list[dict]] = {cat: [] for cat in EXEMPLAR_CATEGORIES}
    for record in records:
        category = record.failure_category
        if category not in EXEMPLAR_CATEGORIES or record.outcome == "success":
            continue
        snippet = extract_failure_snippet(record)
        exemplars[category].append(
            {
                "task_id": record.task_id,
                "run_label": record.run_label,
                "agent_mode": record.agent_mode,
                "failure_category": category,
                "failure_stage": record.failure_stage,
                "failed_step": record.failed_step,
                "failure_message": record.failure_message,
                "snippet": snippet,
            }
        )
    return exemplars


def tag_breakdown(records: list[RunRecord]) -> dict:
    base = failure_breakdown(records)
    by_failure_mode: Counter[str] = Counter()
    by_difficulty: Counter[str] = Counter()
    mode_outcomes: dict[str, dict[str, int]] = defaultdict(lambda: {"success": 0, "failure": 0})

    for record in records:
        mode = record.failure_mode or "untagged"
        diff = record.difficulty or "untagged"
        by_failure_mode[mode] += 1
        by_difficulty[diff] += 1
        bucket = "success" if record.outcome == "success" else "failure"
        mode_outcomes[mode][bucket] += 1

    return {
        **base,
        "by_failure_mode": dict(sorted(by_failure_mode.items())),
        "by_difficulty": dict(sorted(by_difficulty.items())),
        "outcome_by_failure_mode": {k: dict(v) for k, v in sorted(mode_outcomes.items())},
        "category_exemplars": build_category_exemplars(records),
    }


def render_tag_breakdown(breakdown: dict, *, by: str | None = None) -> str:
    if by:
        return _render_breakdown_by_field(breakdown, by)

    lines = [
        "# Failure Breakdown",
        "",
        f"Failed runs: **{breakdown['failed_runs']}**",
        "",
        "## By category",
        "",
    ]
    if breakdown["by_category"]:
        for cat, count in breakdown["by_category"].items():
            lines.append(f"- `{cat}`: {count}")
    else:
        lines.append("- (none)")
    lines.extend(["", "## By stage", ""])
    if breakdown["by_stage"]:
        for stage, count in breakdown["by_stage"].items():
            lines.append(f"- `{stage}`: {count}")
    else:
        lines.append("- (none)")

    lines.extend(["", "## By failure_mode (task tag)", ""])
    for mode, count in breakdown.get("by_failure_mode", {}).items():
        outcomes = breakdown.get("outcome_by_failure_mode", {}).get(mode, {})
        success = outcomes.get("success", 0)
        failure = outcomes.get("failure", 0)
        lines.append(f"- `{mode}`: {count} run(s) — {success} success, {failure} failure")
    lines.extend(["", "## By difficulty (task tag)", ""])
    for diff, count in breakdown.get("by_difficulty", {}).items():
        lines.append(f"- `{diff}`: {count}")

    lines.extend(_render_exemplar_sections(breakdown))

    if breakdown.get("examples"):
        lines.extend(["", "## Examples", ""])
        for ex in breakdown["examples"]:
            lines.extend(_render_example_block(ex))

    return "\n".join(lines) + "\n"


def _render_breakdown_by_field(breakdown: dict, by: str) -> str:
    field = BY_FIELDS.get(by)
    if not field:
        allowed = ", ".join(sorted(BY_FIELDS))
        raise ValueError(f"Unknown --by field {by!r}; choose from: {allowed}")

    title = {
        "failure_mode": "Failure mode (task tag)",
        "difficulty": "Difficulty (task tag)",
        "failure_category": "Failure category",
        "failure_stage": "Failure stage",
    }[by]

    lines = [
        f"# Failure Breakdown — by {by}",
        "",
        f"Failed runs: **{breakdown['failed_runs']}**",
        "",
        f"## By {title}",
        "",
    ]
    counts = breakdown.get(field, {})
    if counts:
        for key, count in counts.items():
            if by == "failure_mode":
                outcomes = breakdown.get("outcome_by_failure_mode", {}).get(key, {})
                lines.append(
                    f"- `{key}`: {count} run(s) — {outcomes.get('success', 0)} success, "
                    f"{outcomes.get('failure', 0)} failure"
                )
            else:
                lines.append(f"- `{key}`: {count}")
    else:
        lines.append("- (none)")

    if by == "failure_category":
        lines.extend(_render_exemplar_sections(breakdown))
    elif breakdown.get("examples"):
        lines.extend(["", "## Examples", ""])
        for ex in breakdown["examples"]:
            if by == "failure_mode":
                continue
            lines.extend(_render_example_block(ex))

    return "\n".join(lines) + "\n"


def _render_exemplar_sections(breakdown: dict) -> list[str]:
    exemplars = breakdown.get("category_exemplars") or {}
    lines: list[str] = []
    for category in EXEMPLAR_CATEGORIES:
        items = exemplars.get(category) or []
        if not items:
            continue
        lines.extend(["", f"## Exemplars — `{category}`", ""])
        for ex in items[:3]:
            label = ex["run_label"]
            lines.append(f"### {ex['task_id']} ({ex['agent_mode']}, {label})")
            lines.append(f"- **Stage:** `{ex['failure_stage']}`")
            lines.append(f"- **Failed step:** {ex['failed_step']}")
            lines.append(f"- **Message:** {ex['failure_message']}")
            if ex.get("snippet"):
                lines.extend(["", "```", ex["snippet"], "```", ""])
    return lines


def _render_example_block(ex: dict) -> list[str]:
    lines = [
        f"### {ex['task_id']} ({ex['agent_mode']})",
        f"- **Category:** `{ex['failure_category']}`",
        f"- **Stage:** `{ex['failure_stage']}`",
        f"- **Failed step:** {ex['failed_step']}",
        f"- **Message:** {ex['failure_message']}",
        "",
    ]
    if ex.get("snippet"):
        lines.extend(["```", ex["snippet"], "```", ""])
    return lines

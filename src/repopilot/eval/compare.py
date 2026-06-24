"""Cross-task and cross-run comparison (Phase 4)."""

from __future__ import annotations

import csv
from io import StringIO
from pathlib import Path

from repopilot.eval.loader import RunRecord, load_task_runs
from repopilot.eval.trajectory_analysis import trajectory_metrics


def comparison_rows(records: list[RunRecord]) -> list[dict]:
    rows: list[dict] = []
    for record in sorted(records, key=lambda r: (r.task_id, r.run_label)):
        traj = trajectory_metrics(record)
        rows.append(_comparison_row(record, traj))
    return rows


def _comparison_row(record: RunRecord, traj: dict) -> dict:
    return {
        "task_id": record.task_id,
        "run_label": record.run_label,
        "agent_mode": record.agent_mode,
        "outcome": record.outcome,
        "tests_passed": record.tests_passed,
        "failure_mode": record.failure_mode or "",
        "difficulty": record.difficulty or "",
        "bug_count": record.bug_count if record.bug_count is not None else "",
        "steps": record.step_count,
        "steps_to_first_edit": traj["steps_to_first_edit"] or "",
        "tests_before_edit": traj["tests_before_edit"],
        "files_touched_count": traj["files_touched_count"],
        "api_calls": record.api_calls,
        "cost": round(record.instance_cost, 4),
        "repair_rounds": record.repair_rounds,
        "started_at": record.started_at or "",
    }


def render_comparison_table_md(rows: list[dict]) -> str:
    if not rows:
        return "_No runs to compare._\n"

    headers = list(rows[0].keys())
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in rows:
        cells = []
        for key in headers:
            val = row[key]
            if val is True:
                cells.append("yes")
            elif val is False:
                cells.append("no")
            elif val is None:
                cells.append("—")
            else:
                cells.append(str(val))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def render_task_run_comparison(records: list[RunRecord], task_id: str) -> str:
    """Compare multiple runs of the same task (latest + history archives)."""
    if not records:
        return f"_No runs found for `{task_id}`._\n"

    rows = comparison_rows(records)
    lines = [
        f"# Task Run Comparison — {task_id}",
        "",
        f"- **Runs compared:** {len(rows)}",
        "",
        "## Runs",
        "",
        render_comparison_table_md(rows).rstrip(),
        "",
    ]

    if len(records) < 2:
        lines.extend(
            [
                "## Multi-run deltas",
                "",
                "_Only one run found. Archive prior runs under "
                f"`runs/{task_id}/history/{{label}}/` to compare across dates or modes._",
                "",
            ]
        )
        return "\n".join(lines)

    baseline = records[0]
    base_traj = trajectory_metrics(baseline)
    lines.extend(["## Multi-run deltas", "", f"Baseline: `{baseline.run_label}` ({baseline.agent_mode})", ""])
    for record in records[1:]:
        traj = trajectory_metrics(record)
        delta_cost = round(record.instance_cost - baseline.instance_cost, 4)
        delta_steps = record.step_count - baseline.step_count
        delta_files = traj["files_touched_count"] - base_traj["files_touched_count"]
        verify = "yes" if record.tests_passed else "no" if record.tests_passed is False else "?"
        base_verify = "yes" if baseline.tests_passed else "no" if baseline.tests_passed is False else "?"
        lines.append(f"### {record.run_label} vs {baseline.run_label}")
        lines.append("")
        lines.append(f"- Outcome: {record.outcome} vs {baseline.outcome}")
        lines.append(f"- Verify: {verify} vs {base_verify}")
        lines.append(f"- Steps: {record.step_count} ({delta_steps:+d})")
        lines.append(f"- Cost: ${record.instance_cost:.4f} ({delta_cost:+.4f})")
        lines.append(f"- Files touched: {traj['files_touched_count']} ({delta_files:+d})")
        lines.append(
            f"- Steps to first edit: {traj['steps_to_first_edit'] or '—'} vs "
            f"{base_traj['steps_to_first_edit'] or '—'}"
        )
        lines.append("")

    return "\n".join(lines)


def render_comparison_report(
    records: list[RunRecord],
    *,
    task_filter: str | None = None,
    highlight_pairs: list[tuple[str, str]] | None = None,
    runs_dir: Path | None = None,
) -> str:
    if task_filter:
        if runs_dir is not None:
            task_records = load_task_runs(runs_dir, task_filter)
            if task_records:
                return render_task_run_comparison(task_records, task_records[0].task_id)
        records = [r for r in records if r.task_id == task_filter or task_filter in r.task_id]
        if len(records) == 1:
            return render_task_run_comparison(records, records[0].task_id)

    rows = comparison_rows(records)
    lines = [
        "# Task Comparison",
        "",
    ]
    if task_filter:
        lines.append(f"- **Filter:** `{task_filter}`")
    lines.append(f"- **Runs compared:** {len(rows)}")
    lines.append("")
    lines.append("## Comparison table")
    lines.append("")
    lines.append(render_comparison_table_md(rows))

    if highlight_pairs:
        lines.extend(["## Paired analysis", ""])
        by_id = {r.task_id: r for r in records}
        for left_id, right_id in highlight_pairs:
            left, right = by_id.get(left_id), by_id.get(right_id)
            if not left or not right:
                continue
            lt, rt = trajectory_metrics(left), trajectory_metrics(right)
            lines.append(f"### {left_id} vs {right_id}")
            lines.append("")
            lines.append(f"- Steps: {left.step_count} vs {right.step_count}")
            lines.append(
                f"- Steps to first edit: {lt['steps_to_first_edit'] or '—'} vs {rt['steps_to_first_edit'] or '—'}"
            )
            lines.append(f"- Files touched: {lt['files_touched_count']} vs {rt['files_touched_count']}")
            lines.append(f"- Cost: ${left.instance_cost:.4f} vs ${right.instance_cost:.4f}")
            lines.append("")

    return "\n".join(lines)


def write_comparison_csv(rows: list[dict], path: Path) -> None:
    if not rows:
        path.write_text("")
        return
    buf = StringIO()
    writer = csv.DictWriter(buf, fieldnames=list(rows[0].keys()))
    writer.writeheader()
    writer.writerows(rows)
    path.write_text(buf.getvalue())


DEFAULT_PAIRS = [
    ("task_002_eval_module", "task_003_expr_multi"),
]

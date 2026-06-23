"""Render evaluation reports from aggregated metrics."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from repopilot.eval.loader import RunRecord, load_all_runs
from repopilot.eval.metrics import aggregate_runs, failure_breakdown, run_records_table


def _render_eval_report(
    records: list[RunRecord],
    metrics: dict,
    breakdown: dict,
    runs_dir: Path,
) -> str:
    lines = [
        "# RepoPilot Evaluation Summary",
        "",
        f"- **Generated:** {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}",
        f"- **Runs directory:** `{runs_dir}`",
        f"- **Total runs:** {metrics['total_runs']}",
        "",
        "## Aggregate metrics",
        "",
        f"- **Success rate (outcome):** {metrics['success_count']}/{metrics['total_runs']} ({metrics['success_rate']}%)",
        f"- **Verify pass rate (runner):** {metrics['verify_pass_count']}/{metrics['total_runs']} ({metrics['verify_pass_rate']}%)",
        f"- **Avg cost:** ${metrics['avg_cost']:.4f}",
        f"- **Avg steps:** {metrics['avg_steps']}",
        f"- **Avg API calls:** {metrics['avg_api_calls']}",
        f"- **Avg repair rounds:** {metrics['avg_repair_rounds']}",
        "",
    ]

    if metrics.get("by_agent_mode"):
        lines.extend(["## By agent mode", "", "| Mode | Runs | Success | Verify pass | Avg cost | Avg steps |", "|------|-----:|--------:|------------:|---------:|----------:|"])
        for mode, stats in metrics["by_agent_mode"].items():
            lines.append(
                f"| {mode} | {stats['runs']} | {stats['success_rate']}% | {stats['verify_pass_rate']}% | "
                f"${stats['avg_cost']:.4f} | {stats['avg_steps']} |"
            )
        lines.append("")

    lines.extend(["## Run table", "", "| task_id | mode | outcome | verify | failure | stage | step | cost |", "|---------|------|---------|--------|---------|-------|-----:|-----:|"])
    for row in run_records_table(records):
        verify = "yes" if row["tests_passed"] is True else "no" if row["tests_passed"] is False else "?"
        lines.append(
            f"| {row['task_id']} | {row['agent_mode']} | {row['outcome']} | {verify} | "
            f"{row['failure_category'] or '—'} | {row['failure_stage'] or '—'} | {row['failed_step'] or '—'} | "
            f"${row['cost']:.4f} |"
        )
    lines.append("")

    if breakdown["failed_runs"]:
        lines.extend(["## Failure breakdown", ""])
        lines.append("### By category")
        lines.append("")
        for cat, count in breakdown["by_category"].items():
            lines.append(f"- `{cat}`: {count}")
        lines.extend(["", "### By stage", ""])
        for stage, count in breakdown["by_stage"].items():
            lines.append(f"- `{stage}`: {count}")
        lines.append("")

    lines.extend(
        [
            "## Artifacts",
            "",
            "| File | Description |",
            "|------|-------------|",
            "| `eval_report.md` | This summary |",
            "| `metrics.json` | Machine-readable aggregate |",
            "| `failure_breakdown.md` | Failure taxonomy details |",
            "| `{task_id}/run_summary.md` | Per-task drill-down |",
            "",
        ]
    )
    return "\n".join(lines)


def _render_failure_breakdown(breakdown: dict) -> str:
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

    if breakdown["examples"]:
        lines.extend(["", "## Examples", ""])
        for ex in breakdown["examples"]:
            lines.append(f"### {ex['task_id']} ({ex['agent_mode']})")
            lines.append(f"- **Category:** `{ex['failure_category']}`")
            lines.append(f"- **Stage:** `{ex['failure_stage']}`")
            lines.append(f"- **Failed step:** {ex['failed_step']}")
            lines.append(f"- **Message:** {ex['failure_message']}")
            lines.append("")

    return "\n".join(lines)


def _render_task_summary(record: RunRecord) -> str:
    lines = [
        f"# Run Summary — {record.task_id}",
        "",
        f"- **Agent mode:** {record.agent_mode}",
        f"- **Model:** {record.model}",
        f"- **Outcome:** {record.outcome}",
        f"- **Runner verify passed:** {record.tests_passed}",
        f"- **Base commit:** `{record.base_commit or 'unknown'}`",
        f"- **Cost:** ${record.instance_cost:.4f} ({record.api_calls} API calls)",
        f"- **Steps:** {record.step_count}",
        "",
    ]

    if record.outcome != "success":
        lines.extend(
            [
                "## Failure attribution",
                "",
                f"- **Category:** `{record.failure_category}`",
                f"- **Stage:** `{record.failure_stage}`",
                f"- **Failed step:** {record.failed_step}",
                f"- **Message:** {record.failure_message}",
                "",
            ]
        )

    if record.pytest_runs:
        lines.extend(["## Pytest runs", ""])
        for run in record.pytest_runs:
            rc = run.get("returncode")
            lines.append(f"- **{run.get('phase')}:** exit {rc} {'✅' if rc == 0 else '❌'}")
        lines.append("")

    if record.steps:
        lines.extend(["## Step timeline", ""])
        for step in record.steps:
            lines.append(f"### Step {step.get('step')}")
            if step.get("reasoning"):
                lines.append(step["reasoning"][:300] + ("…" if len(step["reasoning"]) > 300 else ""))
                lines.append("")
            for i, tc in enumerate(step.get("tool_calls", []), 1):
                preview = tc.get("command", "").replace("\n", " ")[:100]
                lines.append(f"{i}. `[{tc.get('returncode')}]` `{preview}{'…' if len(tc.get('command','')) > 100 else ''}`")
            lines.append("")

    lines.extend(
        [
            "## Source artifacts",
            "",
            f"- Trace: `{record.trace_path}`",
            f"- Meta: `{record.meta_path or 'n/a'}`",
            "",
        ]
    )
    return "\n".join(lines)


def write_eval_summary(
    runs_dir: Path,
    output_dir: Path | None = None,
) -> Path:
    runs_dir = runs_dir.resolve()
    output_dir = (output_dir or runs_dir / "eval" / "summary").resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    records = load_all_runs(runs_dir)
    if not records:
        raise FileNotFoundError(f"No runs with trace.json found under {runs_dir}")

    metrics = aggregate_runs(records)
    breakdown = failure_breakdown(records)

    (output_dir / "eval_report.md").write_text(_render_eval_report(records, metrics, breakdown, runs_dir))
    (output_dir / "metrics.json").write_text(
        json.dumps(
            {
                "generated_at": datetime.now(timezone.utc).isoformat(),
                "runs_dir": str(runs_dir),
                "metrics": metrics,
                "runs": run_records_table(records),
            },
            indent=2,
        )
        + "\n"
    )
    (output_dir / "failure_breakdown.md").write_text(_render_failure_breakdown(breakdown))

    task_eval_root = runs_dir / "eval"
    for record in records:
        task_out = task_eval_root / record.task_id
        task_out.mkdir(parents=True, exist_ok=True)
        (task_out / "run_summary.md").write_text(_render_task_summary(record))

    return output_dir

"""Evaluation harness (Phase 2.5 / 4)."""

from repopilot.eval.compare import comparison_rows, render_comparison_report, write_comparison_csv
from repopilot.eval.failure_analysis import (
    BY_FIELDS,
    EXEMPLAR_CATEGORIES,
    build_category_exemplars,
    extract_failure_snippet,
    render_tag_breakdown,
    tag_breakdown,
)
from repopilot.eval.loader import (
    RunRecord,
    classify_run,
    discover_all_run_paths,
    discover_run_dirs,
    load_all_runs,
    load_run_record,
    load_task_runs,
    resolve_task_run_dir,
)
from repopilot.eval.metrics import aggregate_runs, failure_breakdown, run_records_table
from repopilot.eval.report import write_eval_compare, write_eval_summary
from repopilot.eval.trajectory_analysis import render_trajectory_analysis, trajectory_metrics
from repopilot.eval.visualize import (
    open_run_view,
    render_ascii_bar_chart,
    render_failure_distribution_charts,
    render_mermaid_source,
    render_mermaid_timeline,
    render_run_html,
    write_run_view,
)

__all__ = [
    "BY_FIELDS",
    "EXEMPLAR_CATEGORIES",
    "RunRecord",
    "aggregate_runs",
    "build_category_exemplars",
    "classify_run",
    "comparison_rows",
    "discover_all_run_paths",
    "discover_run_dirs",
    "extract_failure_snippet",
    "failure_breakdown",
    "load_all_runs",
    "load_run_record",
    "load_task_runs",
    "open_run_view",
    "render_ascii_bar_chart",
    "render_comparison_report",
    "render_failure_distribution_charts",
    "render_mermaid_source",
    "render_mermaid_timeline",
    "render_run_html",
    "render_tag_breakdown",
    "render_trajectory_analysis",
    "resolve_task_run_dir",
    "run_records_table",
    "tag_breakdown",
    "trajectory_metrics",
    "write_comparison_csv",
    "write_eval_compare",
    "write_eval_summary",
    "write_run_view",
]

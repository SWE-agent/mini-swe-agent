"""Evaluation harness (Phase 2.5)."""

from repopilot.eval.loader import (
    RunRecord,
    classify_run,
    discover_run_dirs,
    load_all_runs,
    load_run_record,
)
from repopilot.eval.metrics import aggregate_runs, failure_breakdown, run_records_table
from repopilot.eval.report import write_eval_summary

__all__ = [
    "RunRecord",
    "aggregate_runs",
    "classify_run",
    "discover_run_dirs",
    "failure_breakdown",
    "load_all_runs",
    "load_run_record",
    "run_records_table",
    "write_eval_summary",
]

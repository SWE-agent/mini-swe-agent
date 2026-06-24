"""Trace recorder (Phase 2–3)."""

from repopilot.trace.classify import build_failure_reason_md, classify_trace
from repopilot.trace.parse import (
    PytestFailure,
    PytestRun,
    ToolCallRecord,
    TraceStep,
    classify_command_stage,
    extract_files_touched,
    extract_patch_diff,
    extract_pytest_runs,
    extract_viewed_files,
    infer_step_stage,
    is_edit_command,
    iter_tool_rows,
    iter_trace_steps,
    load_trajectory,
    merge_unified_diffs,
    capture_workspace_diff,
    parse_pytest_failures,
)
from repopilot.trace.recorder import TraceArtifacts, TraceContext, build_trace_document, record_trace

__all__ = [
    "PytestFailure",
    "PytestRun",
    "ToolCallRecord",
    "TraceArtifacts",
    "TraceContext",
    "TraceStep",
    "build_failure_reason_md",
    "build_trace_document",
    "capture_workspace_diff",
    "classify_command_stage",
    "classify_trace",
    "extract_files_touched",
    "extract_patch_diff",
    "extract_pytest_runs",
    "extract_viewed_files",
    "infer_step_stage",
    "is_edit_command",
    "iter_tool_rows",
    "iter_trace_steps",
    "load_trajectory",
    "merge_unified_diffs",
    "parse_pytest_failures",
    "record_trace",
]

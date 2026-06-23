"""Trace recorder (Phase 2)."""

from repopilot.trace.parse import (
    PytestRun,
    ToolCallRecord,
    TraceStep,
    extract_patch_diff,
    extract_pytest_runs,
    extract_viewed_files,
    iter_tool_rows,
    iter_trace_steps,
    load_trajectory,
)
from repopilot.trace.recorder import TraceArtifacts, TraceContext, build_trace_document, record_trace

__all__ = [
    "PytestRun",
    "ToolCallRecord",
    "TraceArtifacts",
    "TraceContext",
    "TraceStep",
    "build_trace_document",
    "extract_patch_diff",
    "extract_pytest_runs",
    "extract_viewed_files",
    "iter_tool_rows",
    "iter_trace_steps",
    "load_trajectory",
    "record_trace",
]

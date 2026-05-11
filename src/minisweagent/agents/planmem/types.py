"""Shared types for the PlanMem architecture."""

from dataclasses import dataclass, field
from enum import Enum


@dataclass(frozen=True)
class MemoryStats:
    """Aggregate signals computed from the memory graph.

    Flows memory → planner so the planner can react to "stuck"
    patterns that aren't visible from action/observation alone:
    re-reading the same file repeatedly, repeating identical commands,
    or having explored only a tiny slice of the repo.
    """

    total_nodes: int = 0
    distinct_files_touched: int = 0
    most_touched_file: str | None = None
    most_touched_count: int = 0
    repeat_action_count: int = 0  # max repeats of any single command
    edit_event_count: int = 0
    # Per-file *non-edit* read counts — for saturation detection.
    # Saturation = same file read N+ times without progress.
    file_read_counts: dict[str, int] = field(default_factory=dict)


class TaskPhase(Enum):
    """Agent's current phase in the task lifecycle."""

    EXPLORATION = "exploration"
    HYPOTHESIS = "hypothesis"
    IMPLEMENTATION = "implementation"
    VERIFICATION = "verification"
    BACKTRACK = "backtrack"


@dataclass(frozen=True)
class SubTask:
    """A decomposed unit of work."""

    id: int
    description: str
    phase: TaskPhase
    parent_id: int | None = None
    status: str = "pending"  # pending, active, completed, failed


@dataclass(frozen=True)
class PlanningSignal:
    """Output from the planner that guides the memory controller."""

    current_phase: TaskPhase
    active_subtask: SubTask | None = None
    goal_summary: str = ""
    context_priority_files: list[str] = field(default_factory=list)
    suggested_token_budget: int = 16000
    should_backtrack: bool = False
    goal_drift_detected: bool = False


@dataclass(frozen=True)
class PhaseParams:
    """Phase-specific parameters for the adaptive memory controller."""

    token_budget: int
    diversity_lambda: float
    w_content: float
    w_graph: float
    n_recent: int

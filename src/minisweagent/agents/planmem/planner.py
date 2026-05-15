"""Hierarchical planner with phase detection and goal drift tracking.

The planner has two operating modes:
1. Task start: Uses LLM to decompose the task into sub-tasks (one-time cost).
2. Per-step: Uses free regex/heuristic to detect phase and check goal drift.
"""

import logging
import re
from dataclasses import dataclass, field

from minisweagent.agents.planmem.phase_detector import detect_phase, is_edit_action
from minisweagent.agents.planmem.types import MemoryStats, PlanningSignal, SubTask, TaskPhase

logger = logging.getLogger(__name__)

# Stop words filtered from goal keyword extraction
_STOP_WORDS = frozenset(
    {
        "the",
        "and",
        "for",
        "that",
        "this",
        "with",
        "from",
        "are",
        "was",
        "were",
        "been",
        "have",
        "has",
        "had",
        "not",
        "but",
        "can",
        "will",
        "should",
        "would",
        "could",
        "may",
        "might",
        "shall",
        "does",
        "did",
        "into",
        "than",
        "then",
        "when",
        "where",
        "which",
        "while",
        "also",
        "each",
        "every",
        "all",
        "any",
        "both",
        "few",
        "more",
        "most",
        "other",
        "some",
        "such",
        "only",
        "own",
        "same",
        "too",
        "very",
        "just",
        "because",
        "about",
        "between",
        "through",
        "during",
        "before",
        "after",
        "above",
        "below",
        "here",
        "there",
        "once",
        "further",
        "being",
        "file",
        "line",
        "code",
        "function",
        "class",
        "method",
        "import",
        "return",
        "value",
        "type",
        "name",
        "error",
        "output",
        "input",
        "use",
        "using",
        "used",
        "new",
        "old",
        "first",
        "last",
        "next",
    }
)


@dataclass
class PlannerConfig:
    """Configuration for the hierarchical planner."""

    max_subtasks: int = 8
    decomposition_prompt: str = (
        "Analyze this task and decompose it into 3-6 concrete sub-tasks.\n"
        "Task: {task}\n\n"
        "Output ONLY a numbered list of sub-tasks, one per line.\n"
        "Format: N. [phase] description\n"
        "Phases: exploration, implementation, verification\n"
        "Example:\n"
        "1. [exploration] Find the relevant source files\n"
        "2. [exploration] Reproduce the bug with a test script\n"
        "3. [implementation] Fix the root cause\n"
        "4. [verification] Run the test script to confirm the fix\n"
    )
    exploration_budget: int = 20000
    hypothesis_budget: int = 16000
    implementation_budget: int = 12000
    verification_budget: int = 16000
    backtrack_budget: int = 20000
    default_budget: int = 16000
    drift_threshold: float = 0.15
    drift_window: int = 6
    goal_reminder_cooldown: int = 5
    repeated_edit_threshold: int = 3
    consecutive_failure_threshold: int = 3
    # Memory→planner saturation thresholds
    file_read_saturation_threshold: int = 6  # same file read >=6 times without progress
    repeat_action_threshold: int = 4  # same command issued >=4 times
    replan_cooldown_steps: int = 8  # min steps between two replans


@dataclass
class PlannerState:
    """Mutable state tracked across steps."""

    goal_stack: list[SubTask] = field(default_factory=list)
    completed_subtasks: list[SubTask] = field(default_factory=list)
    failed_subtasks: list[SubTask] = field(default_factory=list)
    current_phase: TaskPhase = TaskPhase.EXPLORATION
    phase_history: list[TaskPhase] = field(default_factory=list)
    goal_keywords: set[str] = field(default_factory=set)
    step_count: int = 0
    last_goal_reminder_step: int = -100
    last_replan_step: int = -(10**6)
    next_subtask_id: int = 0
    file_edit_counts: dict[str, int] = field(default_factory=dict)
    recent_return_codes: list[int] = field(default_factory=list)
    # Per-subtask exit criteria progress
    subtask_progress: dict[int, dict] = field(default_factory=dict)
    # P1b: per-subtask birth message-index. When a sub-task is first adopted,
    # the agent records ``len(self.messages)`` here; on replan/failure we
    # truncate self.messages back to this point (after a safe-cut search).
    subtask_start_msg_idx: dict[int, int] = field(default_factory=dict)


class HierarchicalPlanner:
    """Planner that decomposes tasks and tracks phase + goal drift."""

    def __init__(self, config: PlannerConfig | None = None):
        self.config = config or PlannerConfig()
        self.state = PlannerState()

    def initialize(self, task_description: str, query_fn: object) -> PlanningSignal:
        """One-time setup at task start.

        ``query_fn`` is a callable ``(messages) -> dict`` (typically a model's
        ``query`` method, optionally wrapped by the agent so that calls are
        booked to ``n_calls`` / ``cost``). It can also be the raw model
        object — we'll fall back to ``model.query``.
        """
        self.state.goal_keywords = _extract_keywords(task_description)
        subtasks = self._decompose_with_llm(task_description, query_fn)
        adopted = self._adopt_as_children(subtasks, parent_id=None)
        self.state.goal_stack = list(reversed(adopted))
        if self.state.goal_stack:
            self.state.current_phase = self.state.goal_stack[-1].phase
        logger.info("Planner initialized: %d sub-tasks", len(adopted))
        return self._build_signal()

    def initialize_without_llm(self, task_description: str) -> PlanningSignal:
        """Lightweight init with default sub-tasks (no LLM cost)."""
        self.state.goal_keywords = _extract_keywords(task_description)
        adopted = self._adopt_as_children(_default_subtasks(), parent_id=None)
        self.state.goal_stack = list(reversed(adopted))
        self.state.current_phase = TaskPhase.EXPLORATION
        return self._build_signal()

    def update(
        self,
        action: str,
        observation: str,
        return_code: int,
        thought: str = "",
        memory_stats: MemoryStats | None = None,
    ) -> PlanningSignal:
        """Per-step update. All heuristic, zero LLM cost.

        ``memory_stats`` (optional) carries memory→planner signals — file
        saturation and repeat-action counts that aren't visible from the
        latest action alone. They feed ``_should_backtrack``.
        """
        self.state.step_count += 1
        self.state.recent_return_codes.append(return_code)
        if len(self.state.recent_return_codes) > 10:
            self.state.recent_return_codes = self.state.recent_return_codes[-10:]

        _track_file_edits(self.state, action)
        detected = detect_phase(action, thought, self.state.current_phase)

        should_backtrack = self._should_backtrack(memory_stats)
        if should_backtrack:
            detected = TaskPhase.BACKTRACK

        self.state.current_phase = detected
        self.state.phase_history.append(detected)
        if len(self.state.phase_history) > 20:
            self.state.phase_history = self.state.phase_history[-20:]

        self._record_subtask_progress(action, return_code, detected)
        self._check_subtask_completion(detected, return_code)
        drift = self._detect_goal_drift(action, observation)
        return self._build_signal(goal_drift=drift, should_backtrack=should_backtrack)

    # ── Backtrack detection ─────────────────────────────────────────────────

    def _should_backtrack(self, memory_stats: MemoryStats | None = None) -> bool:
        cfg = self.config
        recent = self.state.recent_return_codes[-cfg.consecutive_failure_threshold :]
        if len(recent) >= cfg.consecutive_failure_threshold and all(rc != 0 for rc in recent):
            logger.info("Backtrack: %d consecutive failures", len(recent))
            return True
        for filepath, count in self.state.file_edit_counts.items():
            if count >= cfg.repeated_edit_threshold:
                logger.info("Backtrack: %s edited %d times", filepath, count)
                return True
        # Memory→planner: file-read saturation & repeat-action.
        if memory_stats is not None:
            saturated = [
                (p, c) for p, c in memory_stats.file_read_counts.items() if c >= cfg.file_read_saturation_threshold
            ]
            if saturated:
                logger.info("Backtrack: read saturation on %s", saturated[0])
                return True
            if memory_stats.repeat_action_count >= cfg.repeat_action_threshold:
                logger.info(
                    "Backtrack: repeat-action count %d",
                    memory_stats.repeat_action_count,
                )
                return True
        return False

    # ── Goal drift detection ────────────────────────────────────────────────

    def _detect_goal_drift(self, action: str, observation: str) -> bool:
        if not self.state.goal_keywords:
            return False
        cfg = self.config
        if self.state.step_count - self.state.last_goal_reminder_step < cfg.goal_reminder_cooldown:
            return False
        recent_text = f"{action} {observation}".lower()
        recent_words = set(re.findall(r"[a-z_][a-z0-9_]{2,}", recent_text))
        if not recent_words:
            return False
        overlap = len(self.state.goal_keywords & recent_words)
        ratio = overlap / len(self.state.goal_keywords)
        if ratio < cfg.drift_threshold and self.state.step_count > cfg.drift_window:
            self.state.last_goal_reminder_step = self.state.step_count
            logger.info("Goal drift: overlap=%.2f < %.2f", ratio, cfg.drift_threshold)
            return True
        return False

    # ── Sub-task management ─────────────────────────────────────────────────

    _PHASE_ORDER = {
        TaskPhase.EXPLORATION: 0,
        TaskPhase.HYPOTHESIS: 1,
        TaskPhase.IMPLEMENTATION: 2,
        TaskPhase.VERIFICATION: 3,
    }

    def _record_subtask_progress(
        self,
        action: str,
        return_code: int,
        detected_phase: TaskPhase,
    ) -> None:
        """Track per-subtask evidence used by the completion predicate."""
        if not self.state.goal_stack:
            return
        active = self.state.goal_stack[-1]
        progress = self.state.subtask_progress.setdefault(
            active.id,
            {"reads": 0, "edits": 0, "verif_pass": 0, "verif_fail": 0},
        )
        if is_edit_action(action):
            progress["edits"] += 1
        elif action and detected_phase == TaskPhase.EXPLORATION:
            progress["reads"] += 1
        if detected_phase == TaskPhase.VERIFICATION:
            if return_code == 0:
                progress["verif_pass"] += 1
            else:
                progress["verif_fail"] += 1

    def _subtask_verified(self, subtask: SubTask, detected_phase: TaskPhase) -> bool:
        """Verification predicate per sub-task phase.

        Replaces the previous "any phase progression = done" heuristic, which
        would mark sub-tasks complete on a single accidental phase flip.
        """
        prog = self.state.subtask_progress.get(subtask.id, {})
        nxt = self._PHASE_ORDER.get(detected_phase, 0)
        cur = self._PHASE_ORDER.get(subtask.phase, 0)
        if nxt <= cur:
            return False  # haven't progressed past this phase
        if subtask.phase == TaskPhase.EXPLORATION:
            return prog.get("reads", 0) >= 1
        if subtask.phase == TaskPhase.HYPOTHESIS:
            return prog.get("reads", 0) + prog.get("edits", 0) >= 1
        if subtask.phase == TaskPhase.IMPLEMENTATION:
            return prog.get("edits", 0) >= 1
        if subtask.phase == TaskPhase.VERIFICATION:
            return prog.get("verif_pass", 0) >= 1
        return True

    def _check_subtask_completion(
        self,
        detected_phase: TaskPhase,
        return_code: int,
    ) -> None:
        if not self.state.goal_stack:
            return
        active = self.state.goal_stack[-1]
        if not self._subtask_verified(active, detected_phase):
            return
        completed = SubTask(
            id=active.id,
            description=active.description,
            phase=active.phase,
            parent_id=active.parent_id,
            status="completed",
        )
        self.state.completed_subtasks.append(completed)
        self.state.goal_stack.pop()
        logger.info("Sub-task completed: %s", completed.description)

    # ── Re-planning ─────────────────────────────────────────────────────────

    def replan_on_backtrack(self, query_fn: object | None = None) -> bool:
        """Close the planning loop on backtrack.

        Marks the active sub-task as failed and pushes new children
        (with ``parent_id`` set, forming a tree) that re-examine the
        situation. Uses ``query_fn`` (counted call) if provided;
        otherwise falls back to a deterministic recovery sequence.
        Cooldown-rate-limited.

        Returns ``True`` if the goal stack was modified.
        """
        if self.state.step_count - self.state.last_replan_step < self.config.replan_cooldown_steps:
            return False
        self.state.last_replan_step = self.state.step_count

        failed = self.state.goal_stack[-1] if self.state.goal_stack else None
        if failed is not None:
            self.state.goal_stack.pop()
            self.state.failed_subtasks.append(
                SubTask(
                    id=failed.id,
                    description=failed.description,
                    phase=failed.phase,
                    parent_id=failed.parent_id,
                    status="failed",
                )
            )
            # Drop the failed sub-task's progress so it can't bleed into
            # the verification predicate of the recovery children.
            self.state.subtask_progress.pop(failed.id, None)
            # NOTE: keep subtask_start_msg_idx[failed.id] until the agent
            # has consumed it for the trajectory rewind (it pops the entry
            # after applying the cut). Do not pop here.
            logger.info("Replan: marking sub-task failed: %s", failed.description)

        children = self._generate_replan_children(failed, query_fn)
        for child in reversed(children):
            self.state.goal_stack.append(child)
        # Reset cumulative stuck signals so the recovery sub-tasks aren't
        # pre-judged as stuck by edits/failures from the previous attempt.
        # (Codex flagged this as a long-task false-positive risk.)
        self.state.file_edit_counts = {}
        self.state.recent_return_codes = []
        logger.info("Replan: pushed %d recovery sub-tasks", len(children))
        return True

    def _generate_replan_children(
        self,
        failed: SubTask | None,
        query_fn: object | None,
    ) -> list[SubTask]:
        """Build recovery sub-tasks; LLM if available, deterministic fallback otherwise."""
        if query_fn is not None and failed is not None:
            try:
                return self._llm_replan(failed, query_fn)
            except Exception:
                logger.warning("LLM replan failed, using fallback", exc_info=True)
        return self._fallback_replan(failed)

    def _llm_replan(self, failed: SubTask, query_fn: object) -> list[SubTask]:
        recent_codes = ",".join(str(rc) for rc in self.state.recent_return_codes[-5:])
        repeated_files = sorted(
            self.state.file_edit_counts,
            key=lambda f: self.state.file_edit_counts[f],
            reverse=True,
        )[:3]
        prompt = (
            "You are a recovery planner. The agent failed sub-task:\n"
            f"  {failed.description} (phase={failed.phase.value})\n"
            f"Recent return codes: [{recent_codes}]\n"
            f"Repeatedly edited files: {repeated_files}\n\n"
            "Output ONLY a numbered list (2-3 lines) of recovery sub-tasks.\n"
            "Format: N. [phase] description\n"
            "Phases: exploration, hypothesis, implementation, verification\n"
            "First step should re-examine assumptions before re-implementing.\n"
        )
        messages = [
            {"role": "system", "content": "You are a task recovery planner."},
            {"role": "user", "content": prompt},
        ]
        response = _call_query_fn(query_fn, messages)
        children = _parse_subtasks(response.get("content", ""), self.config.max_subtasks)
        return self._adopt_as_children(children, parent_id=failed.id)

    def _fallback_replan(self, failed: SubTask | None) -> list[SubTask]:
        parent_id = failed.id if failed is not None else None
        recovery = [
            SubTask(0, "Re-examine assumptions: what did we miss?", TaskPhase.HYPOTHESIS),
            SubTask(0, "Re-explore relevant files / failure context", TaskPhase.EXPLORATION),
            SubTask(0, "Try a different fix approach", TaskPhase.IMPLEMENTATION),
            SubTask(0, "Verify with tests", TaskPhase.VERIFICATION),
        ]
        return self._adopt_as_children(recovery, parent_id=parent_id)

    def record_subtask_birth(self, subtask_id: int, msg_idx: int) -> None:
        """Register the message index at which ``subtask_id`` became active.

        Called by the agent after adoption (initial decomposition or replan)
        so that on later failure the agent can rewind ``self.messages`` to
        this point. Cheap dict write; safe to call repeatedly (idempotent).
        """
        self.state.subtask_start_msg_idx[subtask_id] = msg_idx

    def get_subtask_birth_msg_idx(self, subtask_id: int) -> int | None:
        return self.state.subtask_start_msg_idx.get(subtask_id)

    def _adopt_as_children(
        self,
        subtasks: list[SubTask],
        parent_id: int | None,
    ) -> list[SubTask]:
        """Re-id sub-tasks using planner state's id allocator and set parent_id."""
        adopted = []
        for st in subtasks:
            new_id = self.state.next_subtask_id
            self.state.next_subtask_id += 1
            adopted.append(
                SubTask(
                    id=new_id,
                    description=st.description,
                    phase=st.phase,
                    parent_id=parent_id,
                    status="pending",
                )
            )
        return adopted

    # ── LLM decomposition ───────────────────────────────────────────────────

    def _decompose_with_llm(self, task: str, query_fn: object) -> list[SubTask]:
        prompt = self.config.decomposition_prompt.format(task=task[:3000])
        try:
            messages = [
                {"role": "system", "content": "You are a task decomposition assistant."},
                {"role": "user", "content": prompt},
            ]
            response = _call_query_fn(query_fn, messages)
            return _parse_subtasks(response.get("content", ""), self.config.max_subtasks)
        except Exception:
            logger.warning("LLM decomposition failed, using defaults", exc_info=True)
            return _default_subtasks()

    # ── Signal building ─────────────────────────────────────────────────────

    def _build_signal(
        self,
        goal_drift: bool = False,
        should_backtrack: bool = False,
    ) -> PlanningSignal:
        phase = self.state.current_phase
        budget = self._phase_budget(phase)
        priority_files = sorted(
            self.state.file_edit_counts.keys(),
            key=lambda f: self.state.file_edit_counts[f],
            reverse=True,
        )[:5]
        active = self.state.goal_stack[-1] if self.state.goal_stack else None
        goal_parts = []
        if active:
            goal_parts.append(f"Current sub-task: {active.description}")
        if self.state.goal_keywords:
            goal_parts.append(f"Key terms: {', '.join(sorted(self.state.goal_keywords)[:15])}")
        return PlanningSignal(
            current_phase=phase,
            active_subtask=active,
            goal_summary=" | ".join(goal_parts),
            context_priority_files=priority_files,
            suggested_token_budget=budget,
            should_backtrack=should_backtrack,
            goal_drift_detected=goal_drift,
        )

    def _phase_budget(self, phase: TaskPhase) -> int:
        cfg = self.config
        return {
            TaskPhase.EXPLORATION: cfg.exploration_budget,
            TaskPhase.HYPOTHESIS: cfg.hypothesis_budget,
            TaskPhase.IMPLEMENTATION: cfg.implementation_budget,
            TaskPhase.VERIFICATION: cfg.verification_budget,
            TaskPhase.BACKTRACK: cfg.backtrack_budget,
        }.get(phase, cfg.default_budget)

    @property
    def active_subtask(self) -> SubTask | None:
        return self.state.goal_stack[-1] if self.state.goal_stack else None

    @property
    def progress(self) -> str:
        done = len(self.state.completed_subtasks)
        failed = len(self.state.failed_subtasks)
        pending = len(self.state.goal_stack)
        total = done + failed + pending
        return f"{done}/{total} sub-tasks done, {failed} failed, phase={self.state.current_phase.value}"


# ── Module-level helpers ────────────────────────────────────────────────────


def _call_query_fn(query_fn: object, messages: list[dict]) -> dict:
    """Invoke an LLM query callable or a model object's ``query`` uniformly.

    Lets the planner stay agnostic of whether the agent gave us a counted
    wrapper or the raw model. Returning a dict with at least a ``content`` key
    is enough.
    """
    if callable(query_fn) and not hasattr(query_fn, "query"):
        return query_fn(messages)
    return query_fn.query(messages)


def _extract_keywords(text: str) -> set[str]:
    """Extract meaningful keywords from task text for drift detection."""
    words = set(re.findall(r"[a-z_][a-z0-9_]{2,}", text.lower()))
    return words - _STOP_WORDS


def _track_file_edits(state: PlannerState, action: str) -> None:
    """Track file edit counts for backtrack detection."""
    if not is_edit_action(action):
        return
    first_line = action.split("\n")[0].strip()
    for part in first_line.split():
        part = part.strip("'\"")
        if "/" in part or "." in part:
            if len(part) > 2 and not part.startswith("-"):
                state.file_edit_counts[part] = state.file_edit_counts.get(part, 0) + 1


def _parse_subtasks(text: str, max_subtasks: int) -> list[SubTask]:
    """Parse numbered sub-task list from LLM output."""
    phase_map = {
        "exploration": TaskPhase.EXPLORATION,
        "hypothesis": TaskPhase.HYPOTHESIS,
        "implementation": TaskPhase.IMPLEMENTATION,
        "verification": TaskPhase.VERIFICATION,
    }
    subtasks: list[SubTask] = []

    # Try structured format first: "N. [phase] description"
    for i, match in enumerate(re.finditer(r"^\s*\d+\.\s*\[(\w+)\]\s*(.+)$", text, re.MULTILINE)):
        phase = phase_map.get(match.group(1).lower().strip(), TaskPhase.EXPLORATION)
        subtasks.append(SubTask(id=i, description=match.group(2).strip(), phase=phase))

    # Fallback: plain numbered list with phase inference
    if not subtasks:
        for i, match in enumerate(re.finditer(r"^\s*\d+\.\s*(.+)$", text, re.MULTILINE)):
            desc = match.group(1).strip()
            desc_lower = desc.lower()
            # Order matters: verification before implementation, because
            # "Verify the fix" contains both "verify" and "fix" — the verb
            # at the front is the actual phase.
            if any(kw in desc_lower for kw in ("find", "read", "search", "explore", "understand", "reproduce")):
                phase = TaskPhase.EXPLORATION
            elif any(kw in desc_lower for kw in ("test", "verify", "run", "check", "confirm")):
                phase = TaskPhase.VERIFICATION
            elif any(kw in desc_lower for kw in ("fix", "edit", "change", "modify", "implement", "add", "remove")):
                phase = TaskPhase.IMPLEMENTATION
            else:
                phase = TaskPhase.EXPLORATION
            subtasks.append(SubTask(id=i, description=desc, phase=phase))

    return subtasks[:max_subtasks] if subtasks else _default_subtasks()


def _default_subtasks() -> list[SubTask]:
    """Fallback sub-task sequence."""
    return [
        SubTask(0, "Find and read relevant source files", TaskPhase.EXPLORATION),
        SubTask(1, "Reproduce the issue", TaskPhase.EXPLORATION),
        SubTask(2, "Identify root cause", TaskPhase.HYPOTHESIS),
        SubTask(3, "Implement the fix", TaskPhase.IMPLEMENTATION),
        SubTask(4, "Verify the fix", TaskPhase.VERIFICATION),
    ]

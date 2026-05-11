"""PlanMemAgent: Hierarchical planning + adaptive memory co-design.

Extends MemorySearchAgent with:
1. HierarchicalPlanner — task decomposition, phase detection, goal drift tracking
2. AdaptiveMemoryController — phase-aware beam search parameter adaptation

The agent overrides query() to insert the planner and adaptive controller
into the existing MemorySearchAgent pipeline.
"""

import logging
import re

from minisweagent.agents.default import LimitsExceeded
from minisweagent.agents.memory_search import MemorySearchAgent, MemorySearchConfig
from minisweagent.agents.planmem.memory_controller import (
    AdaptiveMemoryController,
    MemoryControllerConfig,
)
from minisweagent.agents.planmem.planner import HierarchicalPlanner, PlannerConfig
from minisweagent.agents.planmem.types import MemoryStats, PlanningSignal

logger = logging.getLogger(__name__)


class PlanMemConfig(MemorySearchConfig):
    """Configuration for PlanMemAgent (Pydantic, inherits all parent knobs)."""

    # Planner config
    planner_max_subtasks: int = 8
    planner_drift_threshold: float = 0.15
    planner_drift_window: int = 6
    planner_consecutive_failure_threshold: int = 3
    planner_repeated_edit_threshold: int = 3
    planner_exploration_budget: int = 20000
    planner_hypothesis_budget: int = 16000
    planner_implementation_budget: int = 12000
    planner_verification_budget: int = 16000
    planner_backtrack_budget: int = 20000

    # Memory controller config
    memory_file_boost_factor: float = 1.5
    memory_goal_reminder_interval: int = 5

    # Feature flags for ablation
    enable_planner: bool = True
    enable_adaptive_memory: bool = True
    enable_replanning: bool = True       # close planning loop on backtrack
    enable_memory_to_planner: bool = True  # feed MemoryStats into planner
    use_llm_decomposition: bool = True   # False = use default sub-tasks (cheaper)
    use_llm_replan: bool = True          # False = deterministic recovery sub-tasks

    # P0: planning header injected into system prompt — bounded, cache-friendly.
    # When enabled, append "Phase: ... / Current goal: ..." to messages[0] only
    # when the (phase, active_subtask_id) tuple changes. Strict char cap so it
    # cannot push the model off the submit protocol like the repo card did.
    enable_planning_header: bool = False
    planning_header_max_chars: int = 200


class PlanMemAgent(MemorySearchAgent):
    """Agent with hierarchical planning and adaptive memory co-design.

    Inherits all MemorySearchAgent functionality (memory graph, beam search,
    repo background card, file graph analysis) and adds:
    - Per-step phase detection (heuristic, zero cost)
    - Task decomposition into sub-tasks (LLM, one-time cost at start)
    - Phase-aware beam search parameter adaptation
    - Goal drift detection and reminder injection
    """

    def __init__(self, model, env, *, config_class: type = PlanMemConfig, **kwargs):
        super().__init__(model, env, config_class=config_class, **kwargs)

        # Build sub-component configs from flat PlanMemConfig
        planner_config = PlannerConfig(
            max_subtasks=self.config.planner_max_subtasks,
            drift_threshold=self.config.planner_drift_threshold,
            drift_window=self.config.planner_drift_window,
            consecutive_failure_threshold=self.config.planner_consecutive_failure_threshold,
            repeated_edit_threshold=self.config.planner_repeated_edit_threshold,
            exploration_budget=self.config.planner_exploration_budget,
            hypothesis_budget=self.config.planner_hypothesis_budget,
            implementation_budget=self.config.planner_implementation_budget,
            verification_budget=self.config.planner_verification_budget,
            backtrack_budget=self.config.planner_backtrack_budget,
        )
        memory_config = MemoryControllerConfig(
            file_boost_factor=self.config.memory_file_boost_factor,
            goal_reminder_interval=self.config.memory_goal_reminder_interval,
        )

        self.planner = HierarchicalPlanner(planner_config)
        self.memory_controller = AdaptiveMemoryController(memory_config, self)
        self._planning_signal: PlanningSignal | None = None
        self._initialized: bool = False
        # Last (phase, subtask_id) tuple injected — used to skip re-injection
        # when the planner state is unchanged, preserving prompt cache.
        self._last_header_state: tuple[str, int | None] | None = None

    def run(self, task: str, **kwargs) -> tuple[str, str]:
        """Override run to initialize planner at task start."""
        self._initialized = False
        self._planning_signal = None

        # Initialize planner before the main loop starts
        if self.config.enable_planner:
            if self.config.use_llm_decomposition:
                self._planning_signal = self.planner.initialize(
                    task, self._accounted_query,
                )
            else:
                self._planning_signal = self.planner.initialize_without_llm(task)
            logger.info("Planner: %s", self.planner.progress)

        self._initialized = True
        return super().run(task, **kwargs)

    def _accounted_query(self, messages: list[dict]) -> dict:
        """Model-query wrapper that books the call to the agent's ledger.

        Used by planner LLM calls (decomposition / replan) so their cost
        is counted against ``cost_limit`` / ``step_limit`` like any other
        LLM call.
        """
        self.n_calls += 1
        response = self.model.query(messages)
        self.cost += response.get("extra", {}).get("cost", 0.0)
        return response

    def query(self) -> dict:
        """Override query to use adaptive memory controller.

        Flow:
        1. Check limits (using agent counters, same contract as DefaultAgent)
        2. Update planner with latest action/observation (heuristic, free)
        3. Get adaptive parameters from planning signal
        4. Run beam search with adaptive params
        5. Query LLM with selected context (counted via ``_accounted_query``)
        """
        if 0 < self.config.step_limit <= self.n_calls or 0 < self.config.cost_limit <= self.cost:
            raise LimitsExceeded({
                "role": "exit",
                "content": "LimitsExceeded",
                "extra": {"exit_status": "LimitsExceeded", "submission": ""},
            })

        self._ensure_repo_background_card()

        # Update planner based on the most recent action/observation
        if self._initialized and self.config.enable_planner and len(self.memory_graph) > 2:
            self._update_planner()

        # P0: inject the planning header AFTER planner update so phase / sub-task
        # reflect the latest state. Idempotent (skipped when unchanged).
        if self.config.enable_planning_header:
            self._apply_planning_header()

        # Construct memory graph snapshot for telemetry / future use.
        # NOTE: in toolcall mode (LitellmModel et al), the API requires that
        # every assistant `tool_calls` block be followed by matching `tool`
        # observation messages — re-ordering or dropping nodes breaks that
        # contract and the model behaviour collapses (we observed 250 steps
        # of pure exploration with empty patches). So we ONLY rewrite
        # ``self.messages`` when the conversation is text-based; otherwise
        # we keep the full conversation intact and let the planner signal
        # surface only via downstream channels (phase budgets, replan, etc.)
        if self.config.enable_adaptive_memory and self._planning_signal is not None:
            selected_nodes = self.memory_controller.construct_context(
                self._planning_signal, base_agent=self,
            )
        else:
            selected_nodes = self.construct_context_via_search()
        selected_nodes.sort(key=lambda n: n.id)

        toolcall_mode = self._is_toolcall_conversation()
        original_messages = self.messages
        if not toolcall_mode:
            max_chars = self._max_node_chars()
            self.messages = [
                {"role": n.role, "content": self._compress_content(
                    getattr(n, "raw_content", n.content), max_chars,
                )}
                for n in selected_nodes
            ]
        try:
            self.n_calls += 1
            response = self.model.query(self.messages)
        finally:
            if not toolcall_mode:
                self.messages = original_messages

        self.cost += response.get("extra", {}).get("cost", 0.0)
        self.add_messages(response)
        return response

    @staticmethod
    def _is_toolcall_conversation_static(messages: list[dict]) -> bool:
        for m in messages:
            if m.get("role") == "tool" or m.get("tool_calls"):
                return True
        return False

    def _is_toolcall_conversation(self) -> bool:
        return self._is_toolcall_conversation_static(self.messages)

    def _update_planner(self) -> None:
        """Extract action/observation from the last memory nodes and update the planner."""
        last_action = ""
        last_observation = ""
        last_return_code = 0
        last_thought = ""

        for node in reversed(self.memory_graph):
            # Observations are role="user" in textbased mode and role="tool" in toolcall mode.
            if node.role in ("user", "tool") and not last_observation:
                last_observation = node.content
                rc_match = re.search(r"<returncode>(\d+)</returncode>", node.content)
                if rc_match:
                    last_return_code = int(rc_match.group(1))
            elif node.role == "assistant" and not last_action:
                content = getattr(node, "raw_content", node.content)
                # Prefer structured extra.actions (toolcall + textbased), then
                # bash fence (v2 uses mswea_bash_command, v1 used bash).
                md = node.metadata if isinstance(node.metadata, dict) else {}
                actions = (md.get("source_extra") or {}).get("actions") or []
                if actions:
                    first = actions[0]
                    if isinstance(first, dict):
                        last_action = (
                            first.get("action") or first.get("command")
                            or first.get("cmd")
                            or (first.get("arguments") or {}).get("command")
                            or (first.get("arguments") or {}).get("cmd")
                            or ""
                        ).strip()
                    elif isinstance(first, str):
                        last_action = first.strip()
                if not last_action:
                    action_match = re.search(
                        r"```(?:mswea_bash_command|bash)\s*\n(.*?)\n```",
                        content, re.DOTALL,
                    )
                    if action_match:
                        last_action = action_match.group(1).strip()
                thought_match = re.search(
                    r"THOUGHT:\s*(.*?)(?=\n```|\Z)", content, re.DOTALL | re.IGNORECASE,
                )
                if thought_match:
                    last_thought = thought_match.group(1).strip()

            if last_action and last_observation:
                break

        if not (last_action or last_observation):
            return

        # Memory → planner channel: aggregate stats from the memory graph.
        memory_stats = None
        if self.config.enable_memory_to_planner:
            stats_dict = self.compute_memory_stats()
            memory_stats = MemoryStats(**stats_dict)

        self._planning_signal = self.planner.update(
            action=last_action,
            observation=last_observation,
            return_code=last_return_code,
            thought=last_thought,
            memory_stats=memory_stats,
        )
        logger.debug(
            "Planner update: phase=%s, drift=%s, backtrack=%s",
            self._planning_signal.current_phase.value,
            self._planning_signal.goal_drift_detected,
            self._planning_signal.should_backtrack,
        )

        # Close the planning loop: if planner asks for backtrack, regenerate
        # sub-tasks with the failed one as parent. Cooldown lives inside.
        if self.config.enable_replanning and self._planning_signal.should_backtrack:
            replan_qfn = self._accounted_query if self.config.use_llm_replan else None
            if self.planner.replan_on_backtrack(replan_qfn):
                # Surface the new active sub-task in the signal.
                self._planning_signal = self.planner._build_signal(
                    goal_drift=self._planning_signal.goal_drift_detected,
                    should_backtrack=False,  # consumed
                )
                logger.info("Replan triggered: %s", self.planner.progress)

    # ── P0: planning header (system-prompt-level signal channel) ────────────

    PLANMEM_HEADER_BEGIN = "<!-- PLANMEM_HEADER:BEGIN -->"
    PLANMEM_HEADER_END = "<!-- PLANMEM_HEADER:END -->"

    def _apply_planning_header(self) -> None:
        """Idempotently inject a small planning header into ``messages[0]``.

        Design constraints (see plan/2026-05-11-fix-P0-P1-P2-plan.md C-1..C-3):
        - Must NOT break toolcall pairing → only touches system message (idx 0)
        - Must NOT push model off submit protocol → strict char cap, no
          instructional verbs, no file lists, no submit-protocol hints
        - Must be cache-friendly → skip re-write when (phase, subtask_id)
          tuple hasn't changed
        - Always replaces, never appends → re-injection N times = once
        """
        if not self.messages or self.messages[0].get("role") != "system":
            return
        if self._planning_signal is None:
            # Planner disabled — nothing to inject. If a previous header exists,
            # strip it so the system message reverts to baseline.
            self._strip_planning_header()
            return

        phase = self._planning_signal.current_phase.value
        active = self._planning_signal.active_subtask
        subtask_id = active.id if active is not None else None
        state_key = (phase, subtask_id)

        if state_key == self._last_header_state:
            return  # planner state unchanged → cache stays warm

        block = self._build_planning_block(phase, active)
        existing = self.messages[0].get("content") or ""
        stripped = self._strip_block(
            existing, self.PLANMEM_HEADER_BEGIN, self.PLANMEM_HEADER_END,
        )
        new_system = stripped + ("\n\n" if stripped else "") + block
        self._apply_system_content(new_system)
        self._last_header_state = state_key

    def _build_planning_block(self, phase: str, active) -> str:  # noqa: ANN001
        """Build the planning-header marker block.

        Hard rules:
        - Total block ≤ ``planning_header_max_chars``
        - Only phase name + (optional) sub-task description, both truncated
        - NEVER includes: file names, command examples, "do/use/run" verbs,
          submit-protocol references, or anything that could compete with
          the canonical ``echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT`` flow.
        """
        cap = self.config.planning_header_max_chars
        # Reserve room for the wrapping markers + the two label lines.
        wrap_overhead = (
            len(self.PLANMEM_HEADER_BEGIN) + len(self.PLANMEM_HEADER_END) + 32
        )
        body_budget = max(40, cap - wrap_overhead)

        phase_line = f"Phase: {phase[:40]}"
        if active is None:
            goal_line = "Current goal: (planning warm-up)"
        else:
            # Reserve ~half budget for the description; truncate hard.
            remaining = max(20, body_budget - len(phase_line) - 16)
            desc = (active.description or "").strip().replace("\n", " ")[:remaining]
            goal_line = f"Current goal: {desc}"

        block = (
            f"{self.PLANMEM_HEADER_BEGIN}\n"
            f"{phase_line}\n"
            f"{goal_line}\n"
            f"{self.PLANMEM_HEADER_END}"
        )
        # Final hard cap — if any drift makes us oversized, truncate from the
        # goal line. Toleration of one trailing close-marker is preserved.
        if len(block) > cap:
            overrun = len(block) - cap
            shrunk_goal = goal_line[: max(0, len(goal_line) - overrun - 3)] + "..."
            block = (
                f"{self.PLANMEM_HEADER_BEGIN}\n"
                f"{phase_line}\n"
                f"{shrunk_goal}\n"
                f"{self.PLANMEM_HEADER_END}"
            )
        return block

    def _strip_planning_header(self) -> None:
        """Remove an existing planning header from messages[0] (if any)."""
        if not self.messages or self.messages[0].get("role") != "system":
            return
        existing = self.messages[0].get("content") or ""
        stripped = self._strip_block(
            existing, self.PLANMEM_HEADER_BEGIN, self.PLANMEM_HEADER_END,
        )
        if stripped != existing:
            self._apply_system_content(stripped)
        self._last_header_state = None

    @staticmethod
    def _strip_block(text: str, begin: str, end: str) -> str:
        """Remove the first ``begin..end`` block (inclusive) from text."""
        if begin not in text or end not in text:
            return text.rstrip()
        pattern = re.compile(
            re.escape(begin) + r".*?" + re.escape(end), re.DOTALL,
        )
        return pattern.sub("", text).rstrip()

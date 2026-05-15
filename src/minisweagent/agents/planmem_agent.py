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
    enable_replanning: bool = True  # close planning loop on backtrack
    enable_memory_to_planner: bool = True  # feed MemoryStats into planner
    use_llm_decomposition: bool = True  # False = use default sub-tasks (cheaper)
    use_llm_replan: bool = True  # False = deterministic recovery sub-tasks

    # P0: planning header injected into system prompt — bounded, cache-friendly.
    # When enabled, append "Phase: ... / Current goal: ..." to messages[0] only
    # when the (phase, active_subtask_id) tuple changes. Strict char cap so it
    # cannot push the model off the submit protocol like the repo card did.
    enable_planning_header: bool = False
    planning_header_max_chars: int = 200

    # P1a: per-phase sampling routing — non-prompt channel. Per the design,
    # planner's phase signal flows into model.query() kwargs (temperature etc.)
    # instead of (or in addition to) the prompt. Empty dict per phase = no
    # override. Defaults match common ablation choices for code agents.
    enable_phase_sampling: bool = False
    phase_sampling: dict[str, dict] = {
        "exploration": {"temperature": 0.3},
        "hypothesis": {"temperature": 0.1},
        "implementation": {"temperature": 0.0},
        "verification": {"temperature": 0.0},
        "backtrack": {"temperature": 0.4},
    }

    # P1b: trajectory rewind on replan — non-prompt channel that physically
    # truncates self.messages back to a toolcall-safe boundary just before
    # the failed sub-task started, then injects a short reset note so the
    # next model.query sees a coherent conversation tail.
    enable_trajectory_rewind: bool = False
    rewind_reset_message: str = "Previous approach did not work; reconsider with a different angle."


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
        # P1b: pending trajectory-rewind target message index. Set after a
        # successful replan; consumed at the top of the next query().
        self._pending_rewind: int | None = None
        # Set of sub-task ids we've already registered birth_idx for, to
        # avoid double-registration when planner re-emits the same stack.
        self._registered_birth_ids: set[int] = set()

    def run(self, task: str, **kwargs) -> tuple[str, str]:
        """Override run to initialize planner at task start."""
        self._initialized = False
        self._planning_signal = None
        self._pending_rewind = None
        self._registered_birth_ids = set()

        # Initialize planner before the main loop starts
        if self.config.enable_planner:
            if self.config.use_llm_decomposition:
                self._planning_signal = self.planner.initialize(
                    task,
                    self._accounted_query,
                )
            else:
                self._planning_signal = self.planner.initialize_without_llm(task)
            logger.info("Planner: %s", self.planner.progress)
            # P1b: register birth msg-idx for all initial sub-tasks. The
            # parent ``run()`` seeds messages[0]=system + messages[1]=task,
            # so length-2 is the earliest the agent could rewind to. We
            # use that as the birth-idx for all initial sub-tasks.
            self._register_pending_subtask_births(default_idx=2)

        self._initialized = True
        return super().run(task, **kwargs)

    def _register_pending_subtask_births(self, default_idx: int) -> None:
        """Register ``default_idx`` as birth idx for each new sub-task.

        Trust the caller's ``default_idx``: at initial decomposition it
        passes ``len(self.messages)`` (~2), at replan it may pass the
        upcoming rewind target which is intentionally smaller than the
        current message count so recovery sub-tasks share the boundary.

        Called after the planner pushes new sub-tasks. Idempotent: a
        sub-task whose birth was already recorded keeps its original idx.
        """
        if not self.config.enable_planner:
            return
        for st in self.planner.state.goal_stack:
            if st.id not in self._registered_birth_ids:
                self.planner.record_subtask_birth(st.id, default_idx)
                self._registered_birth_ids.add(st.id)

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
            raise LimitsExceeded(
                {
                    "role": "exit",
                    "content": "LimitsExceeded",
                    "extra": {"exit_status": "LimitsExceeded", "submission": ""},
                }
            )

        # P1b: consume a pending rewind BEFORE any other prompt mutation.
        # Order matters: rewind shortens messages, then header/card etc.
        # re-apply on the shortened tail.
        if self.config.enable_trajectory_rewind and self._pending_rewind is not None:
            self._apply_rewind(self._pending_rewind)
            self._pending_rewind = None

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
                self._planning_signal,
                base_agent=self,
            )
        else:
            selected_nodes = self.construct_context_via_search()
        selected_nodes.sort(key=lambda n: n.id)

        toolcall_mode = self._is_toolcall_conversation()
        original_messages = self.messages
        if not toolcall_mode:
            max_chars = self._max_node_chars()
            self.messages = [
                {
                    "role": n.role,
                    "content": self._compress_content(
                        getattr(n, "raw_content", n.content),
                        max_chars,
                    ),
                }
                for n in selected_nodes
            ]

        # P1a: per-phase sampling override. Non-prompt channel — flows into the
        # model API kwargs (temperature et al.), not into the messages list.
        sampling_kwargs = self._phase_sampling_kwargs()
        try:
            self.n_calls += 1
            response = self.model.query(self.messages, **sampling_kwargs)
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
                            first.get("action")
                            or first.get("command")
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
                        content,
                        re.DOTALL,
                    )
                    if action_match:
                        last_action = action_match.group(1).strip()
                thought_match = re.search(
                    r"THOUGHT:\s*(.*?)(?=\n```|\Z)",
                    content,
                    re.DOTALL | re.IGNORECASE,
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
            # Capture failed sub-task id BEFORE replan pops it so we can look
            # up its birth msg-idx for trajectory rewind.
            failed_id_before = self.planner.state.goal_stack[-1].id if self.planner.state.goal_stack else None
            if self.planner.replan_on_backtrack(replan_qfn):
                # P1b: schedule a trajectory rewind to the failed sub-task's
                # birth point. Consumed at top of next query() before the
                # next model call.
                if self.config.enable_trajectory_rewind and failed_id_before is not None:
                    birth = self.planner.get_subtask_birth_msg_idx(failed_id_before)
                    if birth is not None:
                        self._pending_rewind = birth
                    # The birth-idx entry is no longer needed.
                    self.planner.state.subtask_start_msg_idx.pop(
                        failed_id_before,
                        None,
                    )
                # Register birth idx for newly-pushed recovery sub-tasks.
                # Use the (about-to-be-rewound) target index when a rewind
                # is pending so all recovery sub-tasks share the boundary.
                default_idx = self._pending_rewind if self._pending_rewind is not None else len(self.messages)
                self._register_pending_subtask_births(default_idx=default_idx)
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

        existing = self.messages[0].get("content") or ""
        marker_present = self.PLANMEM_HEADER_BEGIN in existing
        # Cache check has two parts:
        # 1. Same planner state as last injection (idempotency)
        # 2. The marker is still present in the system message (external
        #    code paths like _ensure_repo_background_card might have
        #    rewritten messages[0]; in that case we must re-inject)
        if state_key == self._last_header_state and marker_present:
            return  # planner state unchanged AND marker intact → cache warm

        block = self._build_planning_block(phase, active)
        stripped = self._strip_block(
            existing,
            self.PLANMEM_HEADER_BEGIN,
            self.PLANMEM_HEADER_END,
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
        wrap_overhead = len(self.PLANMEM_HEADER_BEGIN) + len(self.PLANMEM_HEADER_END) + 32
        body_budget = max(40, cap - wrap_overhead)

        phase_line = f"Phase: {phase[:40]}"
        if active is None:
            goal_line = "Current goal: (planning warm-up)"
        else:
            # Reserve ~half budget for the description; truncate hard.
            remaining = max(20, body_budget - len(phase_line) - 16)
            # Sanitize: strip the marker substrings so a maliciously- or
            # accidentally-formatted sub-task description from the LLM
            # decomposition cannot break the strip-block regex.
            desc = active.description or ""
            desc = desc.replace(self.PLANMEM_HEADER_BEGIN, "").replace(
                self.PLANMEM_HEADER_END,
                "",
            )
            desc = desc.strip().replace("\n", " ")[:remaining]
            goal_line = f"Current goal: {desc}"

        block = f"{self.PLANMEM_HEADER_BEGIN}\n{phase_line}\n{goal_line}\n{self.PLANMEM_HEADER_END}"
        # Final hard cap — if any drift makes us oversized, truncate from the
        # goal line. Toleration of one trailing close-marker is preserved.
        if len(block) > cap:
            overrun = len(block) - cap
            shrunk_goal = goal_line[: max(0, len(goal_line) - overrun - 3)] + "..."
            block = f"{self.PLANMEM_HEADER_BEGIN}\n{phase_line}\n{shrunk_goal}\n{self.PLANMEM_HEADER_END}"
        return block

    def _strip_planning_header(self) -> None:
        """Remove an existing planning header from messages[0] (if any)."""
        if not self.messages or self.messages[0].get("role") != "system":
            return
        existing = self.messages[0].get("content") or ""
        stripped = self._strip_block(
            existing,
            self.PLANMEM_HEADER_BEGIN,
            self.PLANMEM_HEADER_END,
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
            re.escape(begin) + r".*?" + re.escape(end),
            re.DOTALL,
        )
        return pattern.sub("", text).rstrip()

    # ── P1a: phase-aware sampling routing (non-prompt channel) ──────────────

    def _phase_sampling_kwargs(self) -> dict:
        """Look up sampling kwargs for the current phase.

        Returns a dict like ``{"temperature": 0.0}`` that is passed as kwargs
        to ``self.model.query``. Empty dict means "no override" — the model's
        baseline ``model_kwargs`` (typically ``temperature=0.0``) wins.

        Critical invariants:
        - Returns ``{}`` when feature is disabled or planner gave no signal,
          so default behaviour is byte-identical to baseline
        - Only keys known to litellm's completion API (``temperature``,
          ``top_p``, etc.) — never injects ``messages``/``tools``/etc.
        - Read-only lookup; does not mutate ``self.config.phase_sampling``
        """
        if not self.config.enable_phase_sampling:
            return {}
        if self._planning_signal is None:
            return {}
        phase = self._planning_signal.current_phase.value
        per_phase = self.config.phase_sampling or {}
        kwargs = per_phase.get(phase, {})
        if not isinstance(kwargs, dict):
            return {}
        # Whitelist: only pass through known sampling parameters to keep the
        # surface tight and prevent silent injection of unrelated fields.
        allowed = {"temperature", "top_p", "top_k", "seed", "max_tokens", "presence_penalty", "frequency_penalty"}
        return {k: v for k, v in kwargs.items() if k in allowed}

    # ── P1b: trajectory rewind on replan (non-prompt channel) ───────────────

    def _safe_cut_point(self, target_idx: int) -> int:
        """Return the largest msg index ``<= target_idx`` that is safe to
        truncate at without breaking toolcall ``assistant.tool_calls`` ↔
        ``role: tool`` pairing.

        Safe rule: the resulting prefix ``self.messages[:cut]`` MUST have
        no unmatched ``assistant.tool_calls[id=X]`` (no later ``role:tool``
        ``tool_call_id=X``) within the kept prefix. We walk backward from
        ``target_idx`` and accept the first prefix that satisfies the
        invariant *and* ends on a non-assistant boundary.

        Hard invariants:
        - Returned index ≤ ``target_idx``
        - Returned index ≥ 1 (never strips the system message)
        - ``self.messages[:returned]`` has NO orphan tool_call ids
        - ``self.messages[:returned]`` ends on a non-assistant message OR
          on the system message alone
        """
        if target_idx <= 1:
            return 1
        target_idx = min(target_idx, len(self.messages))
        # Walk backward from target_idx looking for a prefix that is both
        # (a) ending on a user/tool/system boundary AND (b) leaves no
        # orphan tool_calls.
        for cut in range(target_idx, 0, -1):
            last_role = self.messages[cut - 1].get("role") if cut > 0 else "system"
            if last_role == "assistant":
                continue  # cannot end on an assistant turn
            if self._prefix_has_orphan_tool_calls(cut):
                continue  # unresolved tool_call_id in the prefix
            return cut
        return 1

    def _prefix_has_orphan_tool_calls(self, cut: int) -> bool:
        """True iff ``self.messages[:cut]`` has an unmatched tool_call_id.

        Walks the prefix once: every ``assistant.tool_calls[id=X]`` opens
        a pending id; every ``role:tool, tool_call_id=X`` closes one.
        Pending non-empty at the end ⇒ orphan.
        """
        pending: set = set()
        for msg in self.messages[:cut]:
            for tc in msg.get("tool_calls") or []:
                tcid = tc.get("id")
                if tcid is not None:
                    pending.add(tcid)
            if msg.get("role") == "tool":
                pending.discard(msg.get("tool_call_id"))
        return bool(pending)

    def _apply_rewind(self, target_idx: int) -> None:
        """Truncate ``self.messages`` (and memory graph) to a safe cut.

        Then inject a single short user note so the model sees a coherent
        "next turn" prompt rather than a hanging observation. The note is
        configurable (``rewind_reset_message``); intentionally short and
        non-instructional so it cannot drift the submit protocol.
        """
        if not self.messages:
            return
        cut = self._safe_cut_point(target_idx)
        if cut >= len(self.messages):
            return  # nothing to rewind
        dropped = len(self.messages) - cut
        # Truncate parent messages list directly. We bypass any add_messages
        # override here because we are SHRINKING the list, which the
        # observer-style hook isn't designed for.
        del self.messages[cut:]
        # Keep memory_graph aligned: drop the same suffix span.
        if hasattr(self, "memory_graph") and len(self.memory_graph) > cut:
            del self.memory_graph[cut:]
            # next_node_id should not be reset (id space is monotonic for
            # downstream beam-search; we just lose access to nodes we cut).
        logger.info(
            "Rewind: cut at idx %d (dropped %d messages) reset_note=%r",
            cut,
            dropped,
            self.config.rewind_reset_message[:60],
        )
        # Inject the reset note via the standard path so the memory graph
        # is updated consistently.
        self.add_messages(
            {
                "role": "user",
                "content": self.config.rewind_reset_message,
            }
        )

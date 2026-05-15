"""Adaptive memory controller that adjusts beam search parameters based on task phase.

Replaces MemorySearchAgent's fixed-parameter construct_context_via_search() with a
phase-aware version. The planner's PlanningSignal drives parameter selection:
- EXPLORATION: large budget, high diversity, content-weighted
- IMPLEMENTATION: small budget, low diversity, graph-weighted (file-focused)
- VERIFICATION: medium budget, balanced
- BACKTRACK: large budget, highest diversity (re-examine everything)
"""

import logging
import time
from dataclasses import dataclass, field

from minisweagent.agents.memory_search import MemoryNode, MemorySearchAgent
from minisweagent.agents.planmem.types import PhaseParams, PlanningSignal, TaskPhase

logger = logging.getLogger(__name__)

# Default phase parameter table
DEFAULT_PHASE_PARAMS: dict[TaskPhase, PhaseParams] = {
    TaskPhase.EXPLORATION: PhaseParams(
        token_budget=20000,
        diversity_lambda=0.5,
        w_content=0.7,
        w_graph=0.3,
        n_recent=4,
    ),
    TaskPhase.HYPOTHESIS: PhaseParams(
        token_budget=16000,
        diversity_lambda=0.6,
        w_content=0.5,
        w_graph=0.5,
        n_recent=6,
    ),
    TaskPhase.IMPLEMENTATION: PhaseParams(
        token_budget=12000,
        diversity_lambda=0.9,
        w_content=0.3,
        w_graph=0.7,
        n_recent=8,
    ),
    TaskPhase.VERIFICATION: PhaseParams(
        token_budget=16000,
        diversity_lambda=0.7,
        w_content=0.5,
        w_graph=0.5,
        n_recent=6,
    ),
    TaskPhase.BACKTRACK: PhaseParams(
        token_budget=20000,
        diversity_lambda=0.4,
        w_content=0.6,
        w_graph=0.4,
        n_recent=4,
    ),
}


@dataclass
class MemoryControllerConfig:
    """Configuration for the adaptive memory controller."""

    phase_params: dict[str, dict] = field(
        default_factory=lambda: {
            phase.value: {
                "token_budget": p.token_budget,
                "diversity_lambda": p.diversity_lambda,
                "w_content": p.w_content,
                "w_graph": p.w_graph,
                "n_recent": p.n_recent,
            }
            for phase, p in DEFAULT_PHASE_PARAMS.items()
        }
    )
    file_boost_factor: float = 1.5
    goal_reminder_interval: int = 5
    goal_reminder_max_chars: int = 300


class AdaptiveMemoryController:
    """Phase-aware memory controller that adapts beam search parameters.

    This controller wraps MemorySearchAgent's existing beam search machinery,
    overriding parameters based on the current PlanningSignal.
    """

    def __init__(
        self,
        config: MemoryControllerConfig | None = None,
        base_agent: MemorySearchAgent | None = None,
    ):
        self.config = config or MemoryControllerConfig()
        self.base_agent = base_agent
        self._phase_params_cache = self._build_phase_params()
        self._last_goal_reminder_step: int = -100
        self._step_count: int = 0

    def _build_phase_params(self) -> dict[TaskPhase, PhaseParams]:
        """Parse config dict into typed PhaseParams."""
        result = dict(DEFAULT_PHASE_PARAMS)
        for phase_str, params_dict in self.config.phase_params.items():
            try:
                phase = TaskPhase(phase_str)
                result[phase] = PhaseParams(**params_dict)
            except (ValueError, TypeError):
                logger.warning("Invalid phase params for %s, using defaults", phase_str)
        return result

    def get_phase_params(self, phase: TaskPhase) -> PhaseParams:
        """Get parameters for a given phase."""
        return self._phase_params_cache.get(phase, DEFAULT_PHASE_PARAMS[TaskPhase.EXPLORATION])

    def construct_context(
        self,
        planning_signal: PlanningSignal,
        base_agent: MemorySearchAgent | None = None,
    ) -> list[MemoryNode]:
        """Build context using phase-aware beam search.

        This is the main entry point, replacing MemorySearchAgent.construct_context_via_search().

        Args:
            planning_signal: Current planning state from the HierarchicalPlanner.
            base_agent: The MemorySearchAgent instance (overrides self.base_agent).

        Returns:
            List of selected MemoryNode objects for the LLM context.
        """
        agent = base_agent or self.base_agent
        if agent is None:
            raise ValueError("No base_agent provided")

        self._step_count += 1
        params = self.get_phase_params(planning_signal.current_phase)

        # Use suggested budget from planner if available, else phase default
        token_budget = planning_signal.suggested_token_budget or params.token_budget

        logger.info(
            "Adaptive context: phase=%s, budget=%d, lambda=%.2f, w_content=%.2f, w_graph=%.2f, n_recent=%d",
            planning_signal.current_phase.value,
            token_budget,
            params.diversity_lambda,
            params.w_content,
            params.w_graph,
            params.n_recent,
        )

        # Run the beam search with adaptive parameters
        selected = self._beam_search_with_params(
            agent=agent,
            token_budget=token_budget,
            diversity_lambda=params.diversity_lambda,
            w_content=params.w_content,
            w_graph=params.w_graph,
            n_recent=params.n_recent,
            priority_files=planning_signal.context_priority_files,
        )

        # Inject goal reminder if drift detected
        if planning_signal.goal_drift_detected and planning_signal.goal_summary:
            selected = self._inject_goal_reminder(selected, planning_signal.goal_summary)

        return selected

    def _beam_search_with_params(
        self,
        *,
        agent: MemorySearchAgent,
        token_budget: int,
        diversity_lambda: float,
        w_content: float,
        w_graph: float,
        n_recent: int,
        priority_files: list[str],
    ) -> list[MemoryNode]:
        """Run MemorySearchAgent's beam search with overridden parameters.

        Instead of duplicating 200 lines of beam search code, we temporarily
        override the agent's config, run the search, then restore.
        """
        if not agent.memory_graph:
            return []

        # Save original config values
        orig_token_budget = agent.config.token_budget
        orig_diversity_lambda = agent.config.diversity_lambda
        orig_w_content = agent.config.w_content
        orig_w_graph = agent.config.w_graph
        orig_n_recent = agent.config.n_recent_fixed

        # Apply adaptive parameters
        agent.config.token_budget = token_budget
        agent.config.diversity_lambda = diversity_lambda
        agent.config.w_content = w_content
        agent.config.w_graph = w_graph
        agent.config.n_recent_fixed = n_recent

        try:
            selected = agent.construct_context_via_search()
        finally:
            # Restore original config
            agent.config.token_budget = orig_token_budget
            agent.config.diversity_lambda = orig_diversity_lambda
            agent.config.w_content = orig_w_content
            agent.config.w_graph = orig_w_graph
            agent.config.n_recent_fixed = orig_n_recent

        # Apply priority file boosting: re-score and potentially include
        # nodes related to planner-prioritized files
        if priority_files:
            selected = self._boost_priority_files(agent, selected, priority_files, token_budget)

        return selected

    def _boost_priority_files(
        self,
        agent: MemorySearchAgent,
        selected: list[MemoryNode],
        priority_files: list[str],
        token_budget: int,
    ) -> list[MemoryNode]:
        """Ensure nodes touching priority files are included if budget allows.

        The planner identifies files that are critical for the current sub-task.
        If beam search didn't include nodes about these files, try to add them.
        """
        selected_ids = {n.id for n in selected}
        max_chars = token_budget * 4
        current_chars = sum(
            len(agent._compress_content(getattr(n, "raw_content", n.content), agent._max_node_chars()))
            for n in selected
        )

        # Normalize priority file names for matching
        priority_stems = set()
        for f in priority_files:
            stem = f.rstrip("/").rsplit("/", 1)[-1].rsplit(".", 1)[0].lower()
            if stem:
                priority_stems.add(stem)

        if not priority_stems:
            return selected

        # Find unselected nodes that touch priority files
        candidates: list[tuple[int, MemoryNode]] = []
        for node in agent.memory_graph:
            if node.id in selected_ids:
                continue
            node_stems = agent._node_file_stems(node)
            if node_stems & priority_stems:
                node_chars = len(
                    agent._compress_content(getattr(node, "raw_content", node.content), agent._max_node_chars())
                )
                candidates.append((node_chars, node))

        # Sort by recency (higher id = more recent = preferred)
        candidates.sort(key=lambda x: x[1].id, reverse=True)

        # Add candidates within budget
        added = 0
        for node_chars, node in candidates:
            if current_chars + node_chars > max_chars:
                continue
            selected.append(node)
            current_chars += node_chars
            added += 1
            if added >= 3:  # Don't add too many boosted nodes
                break

        if added > 0:
            logger.info("Boosted %d priority-file nodes into context", added)

        return selected

    def _inject_goal_reminder(
        self,
        nodes: list[MemoryNode],
        goal_summary: str,
    ) -> list[MemoryNode]:
        """Insert a synthetic goal-reminder node into the context.

        Placed after the system prompt (node 0) to be visible early.
        """
        # Rate-limit reminders
        if self._step_count - self._last_goal_reminder_step < self.config.goal_reminder_interval:
            return nodes
        self._last_goal_reminder_step = self._step_count

        truncated = goal_summary[: self.config.goal_reminder_max_chars]
        reminder_content = f"GOAL REMINDER: You may be drifting from the original task. Refocus on: {truncated}"

        # Reminder must sort *after* the most-recent observation when the
        # caller re-sorts by id; using id=-1 would leak it ahead of system.
        max_existing_id = max((n.id for n in nodes), default=0)
        reminder = MemoryNode(
            id=max_existing_id + 1,
            role="user",
            content=reminder_content,
            timestamp=time.time(),
            summary=reminder_content[:200],
            raw_content=reminder_content,
        )

        result = list(nodes) + [reminder]

        logger.info("Injected goal reminder into context")
        return result

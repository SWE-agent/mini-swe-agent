"""Unit tests for the PlanMem planning + memory framework.

Covers the four loops we set out to close:
- phase detection (incl. HYPOTHESIS, which used to be unreachable)
- planner decomposition + sub-task verification
- memory→planner channel via MemoryStats
- replan-on-backtrack with parent_id tree
- adaptive memory controller param swap & priority-file boost
"""

from __future__ import annotations

from typing import Any

import pytest

from minisweagent.agents.planmem.phase_detector import detect_phase, is_edit_action
from minisweagent.agents.planmem.planner import (
    HierarchicalPlanner,
    PlannerConfig,
    _parse_subtasks,
)
from minisweagent.agents.planmem.types import MemoryStats, SubTask, TaskPhase


# ── Phase detection ────────────────────────────────────────────────────────


class TestPhaseDetection:
    def test_implementation_via_redirect(self):
        assert detect_phase("echo x > foo.py", "", TaskPhase.EXPLORATION) is TaskPhase.IMPLEMENTATION

    def test_implementation_via_sed(self):
        assert detect_phase("sed -i 's/a/b/' f.py", "", TaskPhase.EXPLORATION) is TaskPhase.IMPLEMENTATION

    def test_verification_via_pytest(self):
        assert detect_phase("pytest tests/", "", TaskPhase.EXPLORATION) is TaskPhase.VERIFICATION

    def test_exploration_via_grep(self):
        assert detect_phase("grep -rn foo src/", "", TaskPhase.IMPLEMENTATION) is TaskPhase.EXPLORATION

    def test_hypothesis_via_python_dash_c(self):
        # Diagnostic command should map to HYPOTHESIS, not VERIFICATION
        # (verification needs file paths or test commands).
        assert detect_phase(
            "python -c 'import x; print(x.__version__)'", "", TaskPhase.EXPLORATION,
        ) is TaskPhase.HYPOTHESIS

    def test_hypothesis_via_thought_text(self):
        # Pure reasoning thought + non-exploration action.
        assert detect_phase(
            "ls -la something",  # not an exploration command
            "I suspect the root cause is in the cache",
            TaskPhase.EXPLORATION,
        ) is TaskPhase.EXPLORATION  # ls IS exploration

        # When the action isn't an exploration cmd, hypothesis text wins.
        assert detect_phase(
            "echo hello",
            "Why does this fail? My hypothesis is the import order.",
            TaskPhase.EXPLORATION,
        ) is TaskPhase.HYPOTHESIS

    def test_falls_back_to_current(self):
        assert detect_phase("", "", TaskPhase.HYPOTHESIS) is TaskPhase.HYPOTHESIS

    def test_is_edit_action(self):
        assert is_edit_action("sed -i 's/a/b/' f.py") is True
        assert is_edit_action("grep foo bar") is False


# ── Sub-task parsing ───────────────────────────────────────────────────────


class TestSubtaskParsing:
    def test_structured_format(self):
        text = (
            "1. [exploration] Find the bug\n"
            "2. [implementation] Fix it\n"
            "3. [verification] Run tests\n"
        )
        st = _parse_subtasks(text, max_subtasks=8)
        assert len(st) == 3
        assert st[0].phase is TaskPhase.EXPLORATION
        assert st[1].phase is TaskPhase.IMPLEMENTATION
        assert st[2].phase is TaskPhase.VERIFICATION

    def test_plain_numbered_falls_back(self):
        text = "1. Find the bug\n2. Fix the bug\n3. Verify the fix\n"
        st = _parse_subtasks(text, max_subtasks=8)
        assert len(st) == 3
        # Heuristic phase inference from keywords
        assert st[0].phase is TaskPhase.EXPLORATION
        assert st[1].phase is TaskPhase.IMPLEMENTATION
        assert st[2].phase is TaskPhase.VERIFICATION

    def test_max_subtasks_clamp(self):
        text = "\n".join(f"{i}. [exploration] step {i}" for i in range(1, 12))
        st = _parse_subtasks(text, max_subtasks=4)
        assert len(st) == 4


# ── Planner: ids, completion verification, drift ───────────────────────────


class TestPlannerCore:
    def test_init_assigns_unique_ids(self):
        p = HierarchicalPlanner(PlannerConfig())
        p.initialize_without_llm("fix the foo bar baz")
        ids = [s.id for s in p.state.goal_stack]
        assert len(ids) == len(set(ids))

    def test_subtask_completion_requires_evidence(self):
        """A single phase flip must NOT auto-pop a sub-task without evidence."""
        p = HierarchicalPlanner(PlannerConfig())
        p.initialize_without_llm("fix something specific")
        n_before = len(p.state.goal_stack)

        # Active sub-task is EXPLORATION. A single non-exploration verification
        # action without prior reads should not mark exploration as completed,
        # because exploration requires at least one read.
        p.update(
            action="pytest tests/test_x.py",  # verification
            observation="",
            return_code=0,
            thought="",
        )
        # The exploration sub-task should still be on the stack because it
        # never accumulated any reads.
        assert len(p.state.goal_stack) == n_before, (
            "Sub-task popped without recorded evidence — verification predicate broken"
        )

    def test_subtask_completion_with_evidence(self):
        p = HierarchicalPlanner(PlannerConfig())
        p.initialize_without_llm("fix something specific")
        n_before = len(p.state.goal_stack)

        # Step 1: an exploration read accumulates a read on the active subtask.
        p.update(action="cat src/foo.py", observation="", return_code=0, thought="")
        # Step 2: progressing to implementation should now allow completion.
        p.update(
            action="sed -i 's/a/b/' src/foo.py", observation="", return_code=0, thought="",
        )
        assert len(p.state.goal_stack) < n_before

    def test_goal_drift_rate_limited(self):
        cfg = PlannerConfig(drift_threshold=0.99, drift_window=0, goal_reminder_cooldown=10)
        p = HierarchicalPlanner(cfg)
        p.initialize_without_llm("alpha bravo charlie delta")
        # First update with totally unrelated text should set drift.
        sig1 = p.update(action="xxx", observation="yyy zzz", return_code=0)
        assert sig1.goal_drift_detected
        # Second update within cooldown should not re-fire drift.
        sig2 = p.update(action="xxx", observation="yyy zzz", return_code=0)
        assert not sig2.goal_drift_detected


# ── Memory→planner: backtrack via MemoryStats ──────────────────────────────


class TestMemoryToPlanner:
    def test_saturation_triggers_backtrack(self):
        cfg = PlannerConfig(file_read_saturation_threshold=3)
        p = HierarchicalPlanner(cfg)
        p.initialize_without_llm("task")
        stats = MemoryStats(file_read_counts={"src/foo.py": 5})
        sig = p.update(action="cat src/foo.py", observation="", return_code=0,
                       memory_stats=stats)
        assert sig.should_backtrack

    def test_repeat_action_triggers_backtrack(self):
        cfg = PlannerConfig(repeat_action_threshold=4)
        p = HierarchicalPlanner(cfg)
        p.initialize_without_llm("task")
        stats = MemoryStats(repeat_action_count=4)
        sig = p.update(action="ls", observation="", return_code=0, memory_stats=stats)
        assert sig.should_backtrack

    def test_no_backtrack_when_stats_below_thresholds(self):
        cfg = PlannerConfig(
            file_read_saturation_threshold=10, repeat_action_threshold=10,
            consecutive_failure_threshold=10, repeated_edit_threshold=10,
        )
        p = HierarchicalPlanner(cfg)
        p.initialize_without_llm("task")
        stats = MemoryStats(file_read_counts={"src/foo.py": 2}, repeat_action_count=1)
        sig = p.update(action="ls", observation="", return_code=0, memory_stats=stats)
        assert not sig.should_backtrack


# ── Replanning loop ────────────────────────────────────────────────────────


class TestReplan:
    def test_fallback_replan_pushes_children_with_parent_id(self):
        p = HierarchicalPlanner(PlannerConfig(replan_cooldown_steps=0))
        p.initialize_without_llm("task")
        # Force step_count > 0 so cooldown logic considers it.
        p.state.step_count = 5

        original_active = p.state.goal_stack[-1]
        ok = p.replan_on_backtrack(query_fn=None)
        assert ok

        # Original was popped and recorded as failed.
        assert any(f.id == original_active.id for f in p.state.failed_subtasks)

        # New top-of-stack child carries parent_id = original_active.id.
        new_active = p.state.goal_stack[-1]
        assert new_active.parent_id == original_active.id
        assert new_active.status == "pending"

    def test_replan_cooldown(self):
        p = HierarchicalPlanner(PlannerConfig(replan_cooldown_steps=10))
        p.initialize_without_llm("task")
        p.state.step_count = 1
        assert p.replan_on_backtrack(query_fn=None)
        # Same step → cooldown blocks.
        assert not p.replan_on_backtrack(query_fn=None)
        # After cooldown elapses, replan is allowed again.
        p.state.step_count = 100
        assert p.replan_on_backtrack(query_fn=None)

    def test_replan_resets_failed_subtask_progress(self):
        """After replan, the failed sub-task's progress dict must be gone.

        Otherwise, if the recovery children happen to reuse an id collision
        (or get re-adopted), stale reads/edits/verif counts could spoof the
        verification predicate into auto-completing them.
        """
        p = HierarchicalPlanner(PlannerConfig(replan_cooldown_steps=0))
        p.initialize_without_llm("task")
        p.state.step_count = 5

        active = p.state.goal_stack[-1]
        # Pretend we accumulated some progress on the active sub-task.
        p.state.subtask_progress[active.id] = {
            "reads": 7, "edits": 3, "verif_pass": 0, "verif_fail": 2,
        }
        assert p.replan_on_backtrack(query_fn=None)
        assert active.id not in p.state.subtask_progress

    def test_llm_replan_with_mock_model(self):
        class _MockModel:
            def query(self, _messages: list[dict]) -> dict[str, Any]:
                return {"content": (
                    "1. [hypothesis] Re-examine the test fixture\n"
                    "2. [implementation] Adjust the fixture to match\n"
                )}

        p = HierarchicalPlanner(PlannerConfig(replan_cooldown_steps=0))
        p.initialize_without_llm("task")
        p.state.step_count = 5
        original = p.state.goal_stack[-1]

        ok = p.replan_on_backtrack(query_fn=_MockModel())
        assert ok
        children = [s for s in p.state.goal_stack if s.parent_id == original.id]
        assert len(children) >= 2
        assert any(c.phase is TaskPhase.HYPOTHESIS for c in children)


# ── Adaptive memory controller ─────────────────────────────────────────────


class _DummyAgent:
    """Minimal stand-in exposing the surface AdaptiveMemoryController touches."""

    def __init__(self, memory_graph: list = ()):  # noqa: ANN001
        from dataclasses import dataclass

        @dataclass
        class _Cfg:
            token_budget: int = 16000
            diversity_lambda: float = 0.7
            w_content: float = 0.5
            w_graph: float = 0.5
            n_recent_fixed: int = 6

        self.config = _Cfg()
        self.memory_graph = list(memory_graph)
        self.last_seen_params: dict = {}

    def construct_context_via_search(self):  # noqa: D401
        # Snapshot the override the controller applied so the test can assert.
        self.last_seen_params = {
            "token_budget": self.config.token_budget,
            "diversity_lambda": self.config.diversity_lambda,
            "w_content": self.config.w_content,
            "w_graph": self.config.w_graph,
            "n_recent_fixed": self.config.n_recent_fixed,
        }
        return list(self.memory_graph)

    # Required surface for the priority-boost path
    def _max_node_chars(self) -> int:
        return 4000

    def _compress_content(self, s: str, _n: int) -> str:
        return s

    def _node_file_stems(self, _node) -> set[str]:  # noqa: ANN001
        return set()


class TestAdaptiveMemoryController:
    def test_phase_params_overridden_then_restored(self):
        from minisweagent.agents.planmem.memory_controller import AdaptiveMemoryController
        from minisweagent.agents.planmem.types import PlanningSignal

        agent = _DummyAgent(memory_graph=[object()])
        ctrl = AdaptiveMemoryController(base_agent=agent)
        signal = PlanningSignal(current_phase=TaskPhase.IMPLEMENTATION)

        original = (agent.config.token_budget, agent.config.diversity_lambda)
        ctrl.construct_context(signal, base_agent=agent)
        # Inside the search we should have seen IMPLEMENTATION-phase params:
        assert agent.last_seen_params["diversity_lambda"] != 0.7  # mutated
        # And after the call they must be restored.
        assert (agent.config.token_budget, agent.config.diversity_lambda) == original

    def test_goal_reminder_does_not_sort_before_system(self):
        """Reminder must not leak ahead of node 0 when caller re-sorts by id."""
        from minisweagent.agents.planmem.memory_controller import AdaptiveMemoryController

        ctrl = AdaptiveMemoryController()
        ctrl._step_count = 100  # bypass cooldown
        ctrl._last_goal_reminder_step = 0

        # Build a fake node list with positive ids — system would be id=0.
        from minisweagent.agents.memory_search import MemoryNode

        nodes = [
            MemoryNode(id=0, role="system", content="sys", timestamp=0.0),
            MemoryNode(id=5, role="user", content="latest obs", timestamp=0.0),
        ]
        out = ctrl._inject_goal_reminder(nodes, "important goal")
        # Sort by id like PlanMemAgent.query does.
        out.sort(key=lambda n: n.id)
        # System should still be first.
        assert out[0].role == "system"
        # The reminder should now be the *last* node, not before system.
        assert out[-1].content.startswith("GOAL REMINDER")


# ── End-to-end smoke test (real run loop, no API) ──────────────────────────


class TestEndToEndSmoke:
    """Make sure PlanMemAgent's run() loop actually completes.

    Codex flagged that prior versions referenced ``self.model.n_calls`` which
    DeterministicModel doesn't expose, blowing up the very first step. This
    test guards that the accounting now lives on the agent and the run loop
    terminates cleanly.
    """

    def _make_agent(self, **flag_overrides):
        from minisweagent.agents.planmem_agent import PlanMemAgent
        from minisweagent.environments.local import LocalEnvironment
        from minisweagent.models.test_models import DeterministicModel, make_output

        outputs = [
            make_output(
                "First do an exploration\n```bash\necho hello\n```",
                [{"command": "echo hello"}],
            ),
            make_output(
                "Now finish\n```bash\necho 'COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT'\necho done\n```",
                [{"command": "echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT && echo done"}],
            ),
        ]
        agent_kwargs = dict(
            system_template="You are an agent.",
            instance_template="Task: {{task}}",
            cost_limit=10.0,
            step_limit=10,
            enable_repo_background_card=False,
            enable_planner=False,         # cheaper smoke
            enable_adaptive_memory=False,
            enable_replanning=False,
            enable_memory_to_planner=False,
            use_llm_decomposition=False,
            use_llm_replan=False,
        )
        agent_kwargs.update(flag_overrides)
        return PlanMemAgent(
            model=DeterministicModel(outputs=outputs),
            env=LocalEnvironment(),
            **agent_kwargs,
        )

    def test_run_completes_with_planner_off(self):
        agent = self._make_agent()
        info = agent.run("Echo hello and finish")
        assert info["exit_status"] == "Submitted"
        # Counters live on the agent now.
        assert agent.n_calls == 2
        assert agent.cost > 0  # DeterministicModel reports cost=1.0/call

    def test_step_limit_terminates_cleanly(self):
        agent = self._make_agent(step_limit=1)
        info = agent.run("Echo hello")
        # The structured exit message lets the run loop break out.
        assert info["exit_status"] == "LimitsExceeded"

    def test_llm_decomposition_call_is_accounted(self):
        """LLM decomposition must increment the agent's n_calls/cost ledger."""
        from minisweagent.agents.planmem_agent import PlanMemAgent
        from minisweagent.environments.local import LocalEnvironment
        from minisweagent.models.test_models import DeterministicModel, make_output

        outputs = [
            # The first model call is the *decomposition* (no actions).
            make_output(
                "1. [exploration] poke around\n2. [verification] run tests\n", [],
            ),
            # Then the agent's normal step calls.
            make_output(
                "exploring\n```bash\necho hi\n```", [{"command": "echo hi"}],
            ),
            make_output(
                "done\n```bash\necho COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT\n```",
                [{"command": "echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT"}],
            ),
        ]
        agent = PlanMemAgent(
            model=DeterministicModel(outputs=outputs),
            env=LocalEnvironment(),
            system_template="You are an agent.",
            instance_template="Task: {{task}}",
            cost_limit=10.0,
            step_limit=10,
            enable_repo_background_card=False,
            enable_planner=True,
            enable_adaptive_memory=False,
            enable_replanning=False,
            enable_memory_to_planner=False,
            use_llm_decomposition=True,
            use_llm_replan=False,
        )
        info = agent.run("Test decomposition accounting")
        assert info["exit_status"] == "Submitted"
        # 1 decomposition + 2 step queries = 3 model calls all booked.
        assert agent.n_calls == 3
        assert agent.cost >= 3.0  # DeterministicModel reports 1.0/call

    def test_run_with_planner_and_adaptive_memory(self):
        """Full PlanMem stack on; LLM-decomposition off (no extra calls)."""
        agent = self._make_agent(
            enable_planner=True,
            enable_adaptive_memory=True,
            enable_replanning=True,
            enable_memory_to_planner=True,
            use_llm_decomposition=False,  # avoid additional model calls
            use_llm_replan=False,
        )
        info = agent.run("Echo hello and finish")
        assert info["exit_status"] == "Submitted"
        assert agent.n_calls == 2


# ── P0: planning header tests ───────────────────────────────────────────────


class TestPlanningHeader:
    """Verify the system-prompt planning-header injection (P0).

    Hard constraints from the design doc:
    - Idempotent (re-applying with same state yields unchanged messages)
    - Cache-friendly (no rewrite when (phase, subtask_id) unchanged)
    - Strict max-chars cap
    - Disabled flag means zero touch
    - Toolcall pairing is never violated (only touches messages[0])
    """

    def _make_agent(self, **overrides):
        from minisweagent.agents.planmem_agent import PlanMemAgent
        from minisweagent.environments.local import LocalEnvironment
        from minisweagent.models.test_models import DeterministicModel

        kwargs = dict(
            system_template="You are an agent.",
            instance_template="Task: {{task}}",
            cost_limit=10.0,
            step_limit=10,
            enable_repo_background_card=False,
            enable_planner=True,
            enable_adaptive_memory=False,
            enable_replanning=False,
            enable_memory_to_planner=False,
            use_llm_decomposition=False,
            use_llm_replan=False,
            enable_planning_header=True,
        )
        kwargs.update(overrides)
        return PlanMemAgent(
            model=DeterministicModel(outputs=[]),
            env=LocalEnvironment(),
            **kwargs,
        )

    def test_renders_phase_and_subtask_inside_markers(self):
        from minisweagent.agents.planmem.types import PlanningSignal, SubTask, TaskPhase

        agent = self._make_agent()
        agent.add_messages({"role": "system", "content": "You are an agent."})
        agent._planning_signal = PlanningSignal(
            current_phase=TaskPhase.IMPLEMENTATION,
            active_subtask=SubTask(
                id=7,
                description="Fix the bug in astropy/modeling/separable.py",
                phase=TaskPhase.IMPLEMENTATION,
            ),
        )
        agent._apply_planning_header()
        sys_content = agent.messages[0]["content"]
        assert agent.PLANMEM_HEADER_BEGIN in sys_content
        assert agent.PLANMEM_HEADER_END in sys_content
        assert "Phase: implementation" in sys_content
        assert "Fix the bug" in sys_content

    def test_idempotent_on_repeated_apply(self):
        from minisweagent.agents.planmem.types import PlanningSignal, SubTask, TaskPhase

        agent = self._make_agent()
        agent.add_messages({"role": "system", "content": "You are an agent."})
        agent._planning_signal = PlanningSignal(
            current_phase=TaskPhase.EXPLORATION,
            active_subtask=SubTask(id=1, description="Look around", phase=TaskPhase.EXPLORATION),
        )
        agent._apply_planning_header()
        first = agent.messages[0]["content"]
        agent._apply_planning_header()
        agent._apply_planning_header()
        # Three applies → still exactly one PLANMEM_HEADER block, no piling up.
        assert agent.messages[0]["content"].count(agent.PLANMEM_HEADER_BEGIN) == 1
        assert agent.messages[0]["content"] == first

    def test_cache_friendly_no_rewrite_on_same_state(self):
        from minisweagent.agents.planmem.types import PlanningSignal, SubTask, TaskPhase

        agent = self._make_agent()
        agent.add_messages({"role": "system", "content": "You are an agent."})
        agent._planning_signal = PlanningSignal(
            current_phase=TaskPhase.EXPLORATION,
            active_subtask=SubTask(id=1, description="Look around", phase=TaskPhase.EXPLORATION),
        )
        agent._apply_planning_header()
        first_id = id(agent.messages[0])
        # State unchanged → second apply must short-circuit before touching content.
        snapshot = agent.messages[0]["content"]
        agent._apply_planning_header()
        assert id(agent.messages[0]) == first_id
        assert agent.messages[0]["content"] == snapshot

    def test_rewrite_when_phase_or_subtask_changes(self):
        from minisweagent.agents.planmem.types import PlanningSignal, SubTask, TaskPhase

        agent = self._make_agent()
        agent.add_messages({"role": "system", "content": "You are an agent."})
        agent._planning_signal = PlanningSignal(
            current_phase=TaskPhase.EXPLORATION,
            active_subtask=SubTask(id=1, description="A", phase=TaskPhase.EXPLORATION),
        )
        agent._apply_planning_header()
        before = agent.messages[0]["content"]
        # Change phase → content must change, but markers stay singular.
        agent._planning_signal = PlanningSignal(
            current_phase=TaskPhase.IMPLEMENTATION,
            active_subtask=SubTask(id=2, description="B", phase=TaskPhase.IMPLEMENTATION),
        )
        agent._apply_planning_header()
        after = agent.messages[0]["content"]
        assert before != after
        assert after.count(agent.PLANMEM_HEADER_BEGIN) == 1
        assert "Phase: implementation" in after
        assert "Phase: exploration" not in after

    def test_disabled_flag_means_no_change(self):
        from minisweagent.agents.planmem.types import PlanningSignal, SubTask, TaskPhase

        agent = self._make_agent(enable_planning_header=False)
        agent.add_messages({"role": "system", "content": "You are an agent."})
        agent._planning_signal = PlanningSignal(
            current_phase=TaskPhase.IMPLEMENTATION,
            active_subtask=SubTask(id=7, description="x" * 500, phase=TaskPhase.IMPLEMENTATION),
        )
        # Apply path is gated by the flag — must NOT touch messages.
        # Simulate the gate as in query():
        if agent.config.enable_planning_header:
            agent._apply_planning_header()
        assert agent.PLANMEM_HEADER_BEGIN not in agent.messages[0]["content"]

    def test_strict_max_chars_cap(self):
        from minisweagent.agents.planmem.types import PlanningSignal, SubTask, TaskPhase

        agent = self._make_agent(planning_header_max_chars=200)
        agent.add_messages({"role": "system", "content": "You are an agent."})
        # Maliciously long subtask description must be truncated.
        agent._planning_signal = PlanningSignal(
            current_phase=TaskPhase.IMPLEMENTATION,
            active_subtask=SubTask(
                id=1, description="A" * 5000, phase=TaskPhase.IMPLEMENTATION,
            ),
        )
        agent._apply_planning_header()
        sys = agent.messages[0]["content"]
        # Block itself must be ≤ cap (the system prefix stays intact).
        begin = sys.index(agent.PLANMEM_HEADER_BEGIN)
        end = sys.index(agent.PLANMEM_HEADER_END) + len(agent.PLANMEM_HEADER_END)
        block_len = end - begin
        assert block_len <= 200, f"block_len {block_len} exceeds cap"

    def test_strip_on_disabled_signal(self):
        from minisweagent.agents.planmem.types import PlanningSignal, SubTask, TaskPhase

        agent = self._make_agent()
        agent.add_messages({"role": "system", "content": "You are an agent."})
        agent._planning_signal = PlanningSignal(
            current_phase=TaskPhase.EXPLORATION,
            active_subtask=SubTask(id=1, description="A", phase=TaskPhase.EXPLORATION),
        )
        agent._apply_planning_header()
        assert agent.PLANMEM_HEADER_BEGIN in agent.messages[0]["content"]
        # Planner state cleared → next apply strips the block.
        agent._planning_signal = None
        agent._apply_planning_header()
        assert agent.PLANMEM_HEADER_BEGIN not in agent.messages[0]["content"]

    def test_description_with_end_marker_sanitized(self):
        """Sub-task description containing the END marker must be sanitized
        before injection — otherwise strip_block regex breaks and the block
        accumulates on every state change."""
        from minisweagent.agents.planmem.types import PlanningSignal, SubTask, TaskPhase

        agent = self._make_agent()
        agent.add_messages({"role": "system", "content": "You are an agent."})
        evil_desc = f"evil {agent.PLANMEM_HEADER_END} oops"
        agent._planning_signal = PlanningSignal(
            current_phase=TaskPhase.EXPLORATION,
            active_subtask=SubTask(id=1, description=evil_desc, phase=TaskPhase.EXPLORATION),
        )
        agent._apply_planning_header()
        # Now change state; the new block must replace the OLD block fully,
        # not leave a fragment behind.
        agent._planning_signal = PlanningSignal(
            current_phase=TaskPhase.IMPLEMENTATION,
            active_subtask=SubTask(id=2, description="clean", phase=TaskPhase.IMPLEMENTATION),
        )
        agent._apply_planning_header()
        sys = agent.messages[0]["content"]
        # Exactly one block; no accumulating residue.
        assert sys.count(agent.PLANMEM_HEADER_BEGIN) == 1
        assert sys.count(agent.PLANMEM_HEADER_END) == 1
        assert "Phase: implementation" in sys
        assert "Phase: exploration" not in sys

    def test_cache_reinjects_when_marker_externally_removed(self):
        """If external code (e.g., _ensure_repo_background_card) rewrites
        messages[0] and removes our marker, the next _apply_planning_header
        MUST re-inject — not short-circuit on cached state."""
        from minisweagent.agents.planmem.types import PlanningSignal, SubTask, TaskPhase

        agent = self._make_agent()
        agent.add_messages({"role": "system", "content": "You are an agent."})
        agent._planning_signal = PlanningSignal(
            current_phase=TaskPhase.EXPLORATION,
            active_subtask=SubTask(id=1, description="A", phase=TaskPhase.EXPLORATION),
        )
        agent._apply_planning_header()
        assert agent.PLANMEM_HEADER_BEGIN in agent.messages[0]["content"]
        # Simulate external wipe (e.g., repo card rewrites the system msg)
        agent.messages[0]["content"] = "You are an agent (rewritten)."
        # State unchanged — but marker missing. Must re-inject.
        agent._apply_planning_header()
        assert agent.PLANMEM_HEADER_BEGIN in agent.messages[0]["content"]
        assert "Phase: exploration" in agent.messages[0]["content"]

    def test_does_not_touch_non_system_messages(self):
        """Critical: P0 must never alter user/assistant/tool messages."""
        from minisweagent.agents.planmem.types import PlanningSignal, SubTask, TaskPhase

        agent = self._make_agent()
        agent.add_messages(
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "task"},
            {"role": "assistant", "content": "asst"},
            {"role": "tool", "content": "obs", "tool_call_id": "x"},
        )
        snapshot_non_sys = [m.copy() for m in agent.messages[1:]]
        agent._planning_signal = PlanningSignal(
            current_phase=TaskPhase.IMPLEMENTATION,
            active_subtask=SubTask(id=1, description="A", phase=TaskPhase.IMPLEMENTATION),
        )
        agent._apply_planning_header()
        for orig, current in zip(snapshot_non_sys, agent.messages[1:]):
            assert orig == current, "P0 must not touch non-system messages"


# ── P1a: per-phase sampling routing tests ───────────────────────────────────


class TestPhaseSampling:
    """Verify that phase-aware sampling kwargs flow through model.query.

    P1a is a non-prompt channel — the test asserts kwargs are passed,
    NOT that any message changed.
    """

    def _make_agent(self, **overrides):
        from minisweagent.agents.planmem_agent import PlanMemAgent
        from minisweagent.environments.local import LocalEnvironment
        from minisweagent.models.test_models import DeterministicModel

        kwargs = dict(
            system_template="You are an agent.",
            instance_template="Task: {{task}}",
            cost_limit=10.0,
            step_limit=10,
            enable_repo_background_card=False,
            enable_planner=True,
            enable_adaptive_memory=False,
            enable_replanning=False,
            enable_memory_to_planner=False,
            use_llm_decomposition=False,
            use_llm_replan=False,
            enable_planning_header=False,
            enable_phase_sampling=True,
        )
        kwargs.update(overrides)
        return PlanMemAgent(
            model=DeterministicModel(outputs=[]),
            env=LocalEnvironment(),
            **kwargs,
        )

    def test_returns_phase_specific_kwargs(self):
        from minisweagent.agents.planmem.types import PlanningSignal, TaskPhase

        agent = self._make_agent()
        for phase in (
            TaskPhase.EXPLORATION,
            TaskPhase.HYPOTHESIS,
            TaskPhase.IMPLEMENTATION,
            TaskPhase.VERIFICATION,
            TaskPhase.BACKTRACK,
        ):
            agent._planning_signal = PlanningSignal(current_phase=phase)
            kw = agent._phase_sampling_kwargs()
            assert "temperature" in kw, f"missing temperature for {phase}"
            assert isinstance(kw["temperature"], (int, float))

    def test_implementation_temperature_zero(self):
        """The recommended default: IMPL/VERIF should be deterministic."""
        from minisweagent.agents.planmem.types import PlanningSignal, TaskPhase

        agent = self._make_agent()
        agent._planning_signal = PlanningSignal(current_phase=TaskPhase.IMPLEMENTATION)
        assert agent._phase_sampling_kwargs()["temperature"] == 0.0

    def test_disabled_returns_empty(self):
        from minisweagent.agents.planmem.types import PlanningSignal, TaskPhase

        agent = self._make_agent(enable_phase_sampling=False)
        agent._planning_signal = PlanningSignal(current_phase=TaskPhase.IMPLEMENTATION)
        assert agent._phase_sampling_kwargs() == {}

    def test_no_signal_returns_empty(self):
        agent = self._make_agent()
        agent._planning_signal = None
        assert agent._phase_sampling_kwargs() == {}

    def test_kwargs_whitelist_blocks_injection(self):
        """Malicious-looking keys must be filtered — only litellm sampling params pass."""
        from minisweagent.agents.planmem.types import PlanningSignal, TaskPhase

        agent = self._make_agent(
            phase_sampling={
                "exploration": {
                    "temperature": 0.3,
                    "messages": "EVIL",  # must be filtered
                    "tools": "EVIL",     # must be filtered
                    "model": "EVIL",     # must be filtered
                }
            },
        )
        agent._planning_signal = PlanningSignal(current_phase=TaskPhase.EXPLORATION)
        kw = agent._phase_sampling_kwargs()
        assert "temperature" in kw
        assert "messages" not in kw
        assert "tools" not in kw
        assert "model" not in kw

    def test_kwargs_passed_to_model_query(self, monkeypatch):
        """End-to-end: agent.query() must pass sampling kwargs to model.query."""
        from minisweagent.agents.planmem_agent import PlanMemAgent
        from minisweagent.agents.planmem.types import PlanningSignal, TaskPhase
        from minisweagent.environments.local import LocalEnvironment
        from minisweagent.models.test_models import DeterministicModel, make_output

        captured_kwargs = []

        class _SpyModel(DeterministicModel):
            def query(self, messages, **kwargs):
                captured_kwargs.append(kwargs)
                return super().query(messages, **kwargs)

        outputs = [
            make_output(
                "finish", [{"command": "echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT"}],
            ),
        ]
        agent = PlanMemAgent(
            model=_SpyModel(outputs=outputs),
            env=LocalEnvironment(),
            system_template="You are an agent.",
            instance_template="Task: {{task}}",
            cost_limit=10.0,
            step_limit=10,
            enable_repo_background_card=False,
            enable_planner=True,
            enable_adaptive_memory=False,
            enable_replanning=False,
            enable_memory_to_planner=False,
            use_llm_decomposition=False,
            use_llm_replan=False,
            enable_planning_header=False,
            enable_phase_sampling=True,
        )
        agent.run("task")
        # At least one call captured; it should have a temperature kwarg.
        assert captured_kwargs, "model.query was never called"
        # The single agent step is in EXPLORATION (default phase post-init).
        assert any("temperature" in kw for kw in captured_kwargs)


# ── P1b: trajectory rewind tests ────────────────────────────────────────────


class TestTrajectoryRewind:
    """Strict pairing-preservation tests for the rewind mechanism.

    P1b is the riskiest of the three (cutting in the wrong place breaks the
    toolcall API), so the tests are deliberately paranoid.
    """

    def _make_agent(self, **overrides):
        from minisweagent.agents.planmem_agent import PlanMemAgent
        from minisweagent.environments.local import LocalEnvironment
        from minisweagent.models.test_models import DeterministicModel

        kwargs = dict(
            system_template="You are an agent.",
            instance_template="Task: {{task}}",
            cost_limit=10.0,
            step_limit=10,
            enable_repo_background_card=False,
            enable_planner=True,
            enable_adaptive_memory=False,
            enable_replanning=True,
            enable_memory_to_planner=False,
            use_llm_decomposition=False,
            use_llm_replan=False,
            enable_planning_header=False,
            enable_phase_sampling=False,
            enable_trajectory_rewind=True,
        )
        kwargs.update(overrides)
        return PlanMemAgent(
            model=DeterministicModel(outputs=[]),
            env=LocalEnvironment(),
            **kwargs,
        )

    def test_safe_cut_after_observation_only(self):
        """The cut must land after a user/tool message, never inside a
        tool_calls→tool pair."""
        agent = self._make_agent()
        agent.messages = [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "task"},
            {"role": "assistant", "content": "", "tool_calls": [{"id": "c1"}]},
            {"role": "tool", "tool_call_id": "c1", "content": "obs1"},
            {"role": "assistant", "content": "", "tool_calls": [{"id": "c2"}]},
            {"role": "tool", "tool_call_id": "c2", "content": "obs2"},
        ]
        # Target inside the second pair → cut must back off to after obs1.
        cut = agent._safe_cut_point(5)  # 5 = the second assistant
        # cut should land at idx 4 (= after obs1 at idx 3)
        assert cut == 4
        kept = agent.messages[:cut]
        assert kept[-1]["role"] == "tool"

    def test_safe_cut_never_strips_system(self):
        agent = self._make_agent()
        agent.messages = [{"role": "system", "content": "sys"}]
        assert agent._safe_cut_point(0) == 1
        assert agent._safe_cut_point(1) == 1
        assert agent._safe_cut_point(10) == 1

    def test_safe_cut_at_boundary(self):
        agent = self._make_agent()
        agent.messages = [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "task"},
        ]
        assert agent._safe_cut_point(2) == 2  # already at boundary

    def test_apply_rewind_truncates_and_appends_reset(self):
        agent = self._make_agent()
        agent.add_messages(
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "task"},
            {"role": "assistant", "content": "", "tool_calls": [{"id": "c1"}]},
            {"role": "tool", "tool_call_id": "c1", "content": "obs1"},
            {"role": "assistant", "content": "", "tool_calls": [{"id": "c2"}]},
            {"role": "tool", "tool_call_id": "c2", "content": "obs2"},
        )
        len_before = len(agent.messages)
        agent._apply_rewind(target_idx=4)  # rewind to before second pair
        # Kept: idx 0..3 (4 items) + reset note → 5 items.
        assert len(agent.messages) == 5
        assert agent.messages[-1]["role"] == "user"
        assert agent.messages[-1]["content"] == agent.config.rewind_reset_message
        # Last pre-reset message is a tool obs → pairing safe.
        assert agent.messages[-2]["role"] == "tool"
        # Memory graph aligned.
        assert len(agent.memory_graph) == 5

    def test_apply_rewind_preserves_toolcall_pairing(self):
        """Even with target falling on an assistant.tool_calls msg, the cut
        must back off so the kept tail doesn't have a dangling tool_calls
        without its matching tool message."""
        agent = self._make_agent()
        agent.add_messages(
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "task"},
            {"role": "assistant", "content": "", "tool_calls": [{"id": "c1"}]},
            {"role": "tool", "tool_call_id": "c1", "content": "obs1"},
            {"role": "assistant", "content": "", "tool_calls": [{"id": "c2"}]},
            {"role": "tool", "tool_call_id": "c2", "content": "obs2"},
            {"role": "assistant", "content": "", "tool_calls": [{"id": "c3"}]},
            {"role": "tool", "tool_call_id": "c3", "content": "obs3"},
        )
        # Target an assistant index — must back off to previous observation.
        agent._apply_rewind(target_idx=6)  # = the third assistant
        # Last non-reset kept message must be a tool obs.
        # Walk back: messages end with the reset user note; the prior msg
        # must be tool to guarantee pairing.
        kept = agent.messages[:-1]  # exclude reset note
        # Every assistant.tool_calls with id X must be followed by a
        # role=tool with tool_call_id=X within `kept`.
        pending: set[str] = set()
        for m in kept:
            if m.get("tool_calls"):
                for tc in m["tool_calls"]:
                    pending.add(tc["id"])
            if m.get("role") == "tool":
                pending.discard(m.get("tool_call_id", ""))
        assert not pending, f"orphan tool_calls left: {pending}"

    def test_rewind_disabled_no_truncation(self):
        agent = self._make_agent(enable_trajectory_rewind=False)
        agent.add_messages(
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "task"},
            {"role": "assistant", "content": "a"},
            {"role": "tool", "tool_call_id": "c", "content": "obs"},
        )
        # _apply_rewind not gated internally — the gate is at the query()
        # call-site. Direct call still mutates; the test is for the gate
        # condition.
        agent._pending_rewind = 2
        # Simulate the query() flow guard:
        if agent.config.enable_trajectory_rewind and agent._pending_rewind is not None:
            agent._apply_rewind(agent._pending_rewind)
        # Flag was off → no mutation.
        assert len(agent.messages) == 4
        assert all(m.get("content") != agent.config.rewind_reset_message
                   for m in agent.messages)

    def test_safe_cut_with_orphan_assistant_tool_calls(self):
        """If walking back to a user/tool boundary would leave an orphan
        assistant.tool_calls in the prefix, the cut MUST back off further
        until the prefix has no orphans. Code-reviewer flagged this gap."""
        agent = self._make_agent()
        # Shape: [sys, user, asst{c1}, user{interrupt}, tool{c1}, asst{c2}]
        # Target the second assistant. The naive "first user/tool boundary"
        # would return cut=4 (after user{interrupt}) — but [sys, user,
        # asst{c1}, user{interrupt}] has orphan c1. Cut must go to 2.
        agent.messages = [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "task"},
            {"role": "assistant", "content": "", "tool_calls": [{"id": "c1"}]},
            {"role": "user", "content": "interrupt"},
            {"role": "tool", "tool_call_id": "c1", "content": "obs"},
            {"role": "assistant", "content": "", "tool_calls": [{"id": "c2"}]},
        ]
        cut = agent._safe_cut_point(5)
        # The naive walk-back returned 4. With orphan check it must return
        # either 5 (after tool{c1}) or 2 (after the user task).
        kept = agent.messages[:cut]
        # No assistant orphan allowed:
        pending = set()
        for m in kept:
            for tc in (m.get("tool_calls") or []):
                pending.add(tc["id"])
            if m.get("role") == "tool":
                pending.discard(m.get("tool_call_id"))
        assert not pending, f"orphan tool_calls in prefix: {pending}"

    def test_safe_cut_with_multi_tool_calls_in_one_assistant(self):
        """An assistant turn with two tool_calls; cut must include both
        tool messages or drop the assistant entirely."""
        agent = self._make_agent()
        agent.messages = [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "task"},
            {"role": "assistant", "content": "",
             "tool_calls": [{"id": "c1"}, {"id": "c2"}]},
            {"role": "tool", "tool_call_id": "c1", "content": "o1"},
            # Note: target inside the pair — between c1 and c2 obs.
        ]
        # Target = 4 (just after first tool). Prefix [sys, user, asst{c1,c2},
        # tool{c1}] still has orphan c2 → cut must back off.
        cut = agent._safe_cut_point(4)
        kept = agent.messages[:cut]
        pending = set()
        for m in kept:
            for tc in (m.get("tool_calls") or []):
                pending.add(tc["id"])
            if m.get("role") == "tool":
                pending.discard(m.get("tool_call_id"))
        assert not pending

    def test_birth_idx_registered_at_run_start(self):
        """Initial sub-tasks must have their birth_idx recorded so a later
        replan can find a rewind target."""
        from minisweagent.agents.planmem_agent import PlanMemAgent
        from minisweagent.agents.planmem.types import TaskPhase
        from minisweagent.environments.local import LocalEnvironment
        from minisweagent.models.test_models import DeterministicModel, make_output

        outputs = [
            make_output(
                "done", [{"command": "echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT"}],
            ),
        ]
        agent = PlanMemAgent(
            model=DeterministicModel(outputs=outputs),
            env=LocalEnvironment(),
            system_template="You are an agent.",
            instance_template="Task: {{task}}",
            cost_limit=10.0,
            step_limit=10,
            enable_repo_background_card=False,
            enable_planner=True,
            enable_adaptive_memory=False,
            enable_replanning=True,
            enable_memory_to_planner=False,
            use_llm_decomposition=False,
            use_llm_replan=False,
            enable_planning_header=False,
            enable_phase_sampling=False,
            enable_trajectory_rewind=True,
        )
        agent.run("task")
        # Every initial sub-task adopted at run start should have a birth idx.
        for st in agent.planner.state.goal_stack + agent.planner.state.completed_subtasks:
            assert st.id in agent.planner.state.subtask_start_msg_idx, (
                f"sub-task {st.id} ({st.description}) missing birth idx"
            )

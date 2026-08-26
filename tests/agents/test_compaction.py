import pytest

from minisweagent.agents.compaction import (
    compactable_boundary,
    complete_groups,
    estimate_tokens,
    render_summary_prompt,
)
from minisweagent.agents.default import DefaultAgent
from minisweagent.environments.local import LocalEnvironment
from minisweagent.exceptions import ContextWindowExceeded
from minisweagent.models.test_models import DeterministicModel, make_output


class RecordingModel(DeterministicModel):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.queries = []
        self.summary_queries = []

    def query(self, messages, **kwargs):
        self.queries.append(messages)
        return super().query(messages, **kwargs)

    def generate_text(self, messages, **kwargs):
        self.summary_queries.append(messages)
        return super().generate_text(messages, **kwargs)


class OverflowOnceModel(RecordingModel):
    def query(self, messages, **kwargs):
        if not self.queries:
            self.queries.append(messages)
            raise ContextWindowExceeded
        return super().query(messages, **kwargs)


def make_agent(model, **compaction):
    return DefaultAgent(
        model=model,
        env=LocalEnvironment(),
        system_template="system",
        instance_template="{{ task }}",
        compaction={
            "enabled": True,
            "context_limit": 1_000,
            "keep_tokens": 0,
            "buffer": 100,
            "summary_max_tokens": 100,
            **compaction,
        },
    )


def add_complete_history(agent):
    agent.add_messages(
        {"role": "system", "content": "system"},
        {"role": "user", "content": "task " * 40},
        make_output("first result " * 40, []),
        {"role": "user", "content": "follow-up " * 40},
        make_output("second result " * 40, []),
    )


def test_estimate_tokens_ignores_extra_response_payloads():
    messages = [{"role": "assistant", "content": "done", "extra": {"response": "x" * 100_000}}]

    assert estimate_tokens(messages) < 20


@pytest.mark.parametrize(
    ("compaction", "error"),
    [
        ({"context_limit": 100, "summary_max_tokens": 100}, "summary_max_tokens"),
        ({"context_limit": 1_000, "buffer": 1_000}, "buffer"),
    ],
)
def test_invalid_compaction_budget_is_rejected(compaction, error):
    with pytest.raises(ValueError, match=error):
        make_agent(DeterministicModel(outputs=[]), **compaction)


def test_complete_groups_keep_tool_calls_with_all_outputs():
    messages = [
        {"role": "system", "content": "system"},
        {"role": "user", "content": "task"},
        {"role": "assistant", "extra": {"actions": [{"command": "a"}, {"command": "b"}]}},
        {"role": "tool", "content": "a"},
        {"role": "tool", "content": "b"},
        {"role": "assistant", "content": "next", "extra": {"actions": [{"command": "c"}]}},
        {"role": "tool", "content": "c"},
    ]

    assert complete_groups(messages, 1) == [(1, 5), (5, 7)]
    assert compactable_boundary(messages, 1, 0) == 5


def test_complete_groups_support_responses_api_messages():
    messages = [
        {"type": "message", "role": "system", "content": [{"type": "input_text", "text": "system"}]},
        {"type": "message", "role": "user", "content": [{"type": "input_text", "text": "task"}]},
        {"object": "response", "extra": {"actions": [{"command": "a", "tool_call_id": "call_1"}]}},
        {"type": "function_call_output", "call_id": "call_1", "output": "result"},
        {"object": "response", "extra": {"actions": []}},
    ]

    assert complete_groups(messages, 1) == [(1, 4), (4, 5)]
    assert compactable_boundary(messages, 1, 0) == 4


def test_summary_prompt_includes_previous_summary_and_cross_format_content():
    prompt = render_summary_prompt(
        "Previous: {{ previous_summary }}\nNew: {{ transcript }}",
        "Already inspected a.py",
        [{"type": "function_call_output", "output": "tests passed"}],
    )

    assert "Already inspected a.py" in prompt
    assert "tests passed" in prompt


def test_automatic_compaction_preserves_history_and_shortens_model_input(reset_global_stats):
    model = RecordingModel(
        outputs=[make_output("continue", [], cost=0.25)],
        summary_outputs=["task and first result"],
        cost_per_call=0.25,
    )
    agent = make_agent(model, context_limit=500, buffer=400, summary_max_tokens=100)
    add_complete_history(agent)
    original = list(agent.messages)

    agent.query()

    assert agent.messages[:-1] == original
    assert agent.compaction_state.summary == "task and first result"
    assert agent.n_compaction_calls == 1
    assert agent.cost == pytest.approx(0.5)
    assert len(model.queries[0]) < len(original)
    assert "conversation_summary" in model.queries[0][1]["content"]
    assert agent.serialize()["info"]["model_stats"] == {
        "instance_cost": 0.5,
        "api_calls": 2,
        "task_calls": 1,
        "compaction_calls": 1,
    }


def test_context_overflow_forces_one_compaction_and_retry(reset_global_stats):
    model = OverflowOnceModel(
        outputs=[make_output("recovered", [], cost=0.2)],
        summary_outputs=["checkpoint"],
        cost_per_call=0.2,
    )
    agent = make_agent(model, auto=False)
    add_complete_history(agent)

    assert agent.query()["content"] == "recovered"
    assert len(model.queries) == 2
    assert agent.n_calls == 1
    assert agent.n_compaction_calls == 1
    assert agent.compaction_state.records[0].reason == "overflow"


def test_repeated_compaction_updates_previous_checkpoint(reset_global_stats):
    model = RecordingModel(outputs=[], summary_outputs=["first checkpoint", "updated checkpoint"])
    agent = make_agent(model, context_limit=450, summary_max_tokens=100)
    add_complete_history(agent)
    agent.add_messages(
        {"role": "user", "content": "third request " * 40},
        make_output("third result " * 40, []),
    )

    assert agent._compact_once("auto")
    assert agent._compact_once("auto")
    assert agent.compaction_state.summary == "updated checkpoint"
    assert len(agent.compaction_state.records) == 2
    assert "first checkpoint" in model.summary_queries[1][1]["content"]


def test_empty_checkpoint_is_rejected_without_advancing_state(reset_global_stats):
    model = RecordingModel(outputs=[], summary_outputs=[""], cost_per_call=0.2)
    agent = make_agent(model)
    add_complete_history(agent)

    with pytest.raises(RuntimeError, match="empty checkpoint"):
        agent._compact_once("auto")
    assert agent.compaction_state.summary == ""
    assert agent.compaction_state.compacted_until == 1
    assert agent.n_compaction_calls == 1
    assert agent.cost == 0.2


def test_second_context_overflow_propagates(reset_global_stats):
    class AlwaysOverflowModel(RecordingModel):
        def query(self, messages, **kwargs):
            self.queries.append(messages)
            raise ContextWindowExceeded

    model = AlwaysOverflowModel(outputs=[], summary_outputs=["checkpoint"], cost_per_call=0.2)
    agent = make_agent(model, auto=False)
    add_complete_history(agent)

    with pytest.raises(ContextWindowExceeded):
        agent.query()
    assert len(model.queries) == 2
    assert agent.n_compaction_calls == 1


def test_context_overflow_without_compactable_history_propagates(reset_global_stats):
    model = OverflowOnceModel(outputs=[make_output("unused", [])], summary_outputs=["unused"])
    agent = make_agent(model, auto=False)
    agent.add_messages({"role": "system", "content": "system"}, {"role": "user", "content": "task"})

    with pytest.raises(ContextWindowExceeded):
        agent.query()
    assert len(model.queries) == 1
    assert agent.n_compaction_calls == 0

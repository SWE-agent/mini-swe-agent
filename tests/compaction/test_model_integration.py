"""Tests for compaction integration with LitellmModel and OpenRouterModel.

Verifies that:
- compaction runs inside _prepare_messages_for_api (not on agent's self.messages)
- model config accepts compaction / compaction_kwargs
- the agent's full message list is never mutated
"""

import json
from unittest.mock import MagicMock, patch

import pytest

from minisweagent.compaction.providers.generic.sliding_window import SlidingWindowCompaction
from minisweagent.models.litellm_model import LitellmModel
from minisweagent.models.openrouter_model import OpenRouterModel


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------


def _tc_msg(cmd, role="assistant"):
    """Minimal toolcall-format assistant message."""
    return {
        "role": role,
        "content": None,
        "tool_calls": [
            {
                "id": "call_x",
                "type": "function",
                "function": {"name": "bash", "arguments": json.dumps({"command": cmd})},
            }
        ],
    }


def _tool_msg(output="ok"):
    return {"role": "tool", "content": output, "tool_call_id": "call_x"}


def _build_long_history(n_steps=5):
    """system + user + n_steps × (assistant + tool)."""
    msgs = [
        {"role": "system", "content": "sys", "extra": {}},
        {"role": "user", "content": "task", "extra": {}},
    ]
    for i in range(n_steps):
        msgs.append({**_tc_msg(f"cat file{i}.py"), "extra": {}})
        msgs.append({**_tool_msg(f"content_{i}"), "extra": {}})
    return msgs


def _mock_litellm_response():
    tool_call = MagicMock()
    tool_call.function.name = "bash"
    tool_call.function.arguments = '{"command": "echo done"}'
    tool_call.id = "call_done"
    resp = MagicMock()
    resp.choices = [MagicMock()]
    resp.choices[0].message.tool_calls = [tool_call]
    resp.choices[0].message.model_dump.return_value = {"role": "assistant", "content": None}
    resp.model_dump.return_value = {}
    return resp


# ---------------------------------------------------------------------------
# LitellmModel — config
# ---------------------------------------------------------------------------


class TestLitellmModelCompactionConfig:
    def test_default_compaction_is_none(self):
        model = LitellmModel(model_name="test")
        assert model.config.compaction == "none"

    def test_compaction_strategy_instantiated(self):
        model = LitellmModel(
            model_name="test",
            compaction="generic/sliding_window",
            compaction_kwargs={"keep_last_n_steps": 3},
        )
        assert isinstance(model._compaction, SlidingWindowCompaction)
        assert model._compaction.keep_last_n_steps == 3

    def test_invalid_strategy_raises_on_init(self):
        with pytest.raises(ValueError, match="Unknown compaction strategy"):
            LitellmModel(model_name="test", compaction="no/such/strategy")


# ---------------------------------------------------------------------------
# LitellmModel — _prepare_messages_for_api applies compaction
# ---------------------------------------------------------------------------


class TestLitellmModelPrepareMessages:
    def test_noop_compaction_preserves_all_messages(self):
        model = LitellmModel(model_name="test", compaction="none", set_cache_control=None)
        msgs = _build_long_history(n_steps=3)
        prepared = model._prepare_messages_for_api(msgs)
        # extra stripped → 2 + 3*2 = 8 messages
        assert len(prepared) == 8

    def test_sliding_window_reduces_messages(self):
        model = LitellmModel(
            model_name="test",
            compaction="generic/sliding_window",
            compaction_kwargs={"keep_last_n_steps": 2},
            set_cache_control=None,
        )
        msgs = _build_long_history(n_steps=5)
        prepared = model._prepare_messages_for_api(msgs)
        # anchors(2) + 2 kept steps × 2 messages = 6
        assert len(prepared) == 6

    def test_compaction_does_not_mutate_input(self):
        model = LitellmModel(
            model_name="test",
            compaction="generic/sliding_window",
            compaction_kwargs={"keep_last_n_steps": 1},
            set_cache_control=None,
        )
        msgs = _build_long_history(n_steps=4)
        original_len = len(msgs)
        model._prepare_messages_for_api(msgs)
        assert len(msgs) == original_len

    def test_extra_field_stripped_before_compaction(self):
        """'extra' keys must not appear in prepared messages."""
        model = LitellmModel(model_name="test", compaction="none", set_cache_control=None)
        msgs = _build_long_history(n_steps=2)
        prepared = model._prepare_messages_for_api(msgs)
        for msg in prepared:
            assert "extra" not in msg


# ---------------------------------------------------------------------------
# LitellmModel — compaction applied during real query path (mocked litellm)
# ---------------------------------------------------------------------------


class TestLitellmModelQueryCompaction:
    @patch("minisweagent.models.litellm_model.litellm.completion")
    @patch("minisweagent.models.litellm_model.litellm.cost_calculator.completion_cost")
    def test_query_sends_compacted_messages(self, mock_cost, mock_completion):
        """The messages list sent to litellm.completion is shorter after compaction."""
        mock_completion.return_value = _mock_litellm_response()
        mock_cost.return_value = 0.001

        model = LitellmModel(
            model_name="gpt-4",
            compaction="generic/sliding_window",
            compaction_kwargs={"keep_last_n_steps": 1},
            set_cache_control=None,
        )
        msgs = _build_long_history(n_steps=4)
        model.query(msgs)

        sent_msgs = mock_completion.call_args.kwargs["messages"]
        # anchors(2) + 1 kept step × 2 msgs = 4
        assert len(sent_msgs) == 4

    @patch("minisweagent.models.litellm_model.litellm.completion")
    @patch("minisweagent.models.litellm_model.litellm.cost_calculator.completion_cost")
    def test_agent_messages_not_mutated_after_query(self, mock_cost, mock_completion):
        """The original messages list passed to query() stays intact."""
        mock_completion.return_value = _mock_litellm_response()
        mock_cost.return_value = 0.001

        model = LitellmModel(
            model_name="gpt-4",
            compaction="generic/sliding_window",
            compaction_kwargs={"keep_last_n_steps": 1},
            set_cache_control=None,
        )
        msgs = _build_long_history(n_steps=4)
        original_len = len(msgs)
        model.query(msgs)
        assert len(msgs) == original_len


# ---------------------------------------------------------------------------
# OpenRouterModel — same config fields wired up
# ---------------------------------------------------------------------------


class TestOpenRouterModelCompactionConfig:
    def test_default_compaction_is_none(self):
        model = OpenRouterModel(model_name="test")
        assert model.config.compaction == "none"

    def test_sliding_window_config(self):
        model = OpenRouterModel(
            model_name="test",
            compaction="generic/sliding_window",
            compaction_kwargs={"keep_last_n_steps": 5},
        )
        assert isinstance(model._compaction, SlidingWindowCompaction)
        assert model._compaction.keep_last_n_steps == 5

    def test_prepare_messages_reduces_with_compaction(self):
        model = OpenRouterModel(
            model_name="test",
            compaction="generic/sliding_window",
            compaction_kwargs={"keep_last_n_steps": 2},
            set_cache_control=None,
        )
        msgs = _build_long_history(n_steps=5)
        prepared = model._prepare_messages_for_api(msgs)
        assert len(prepared) == 6  # anchors(2) + 2 steps × 2

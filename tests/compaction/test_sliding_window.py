"""Tests for _group_into_steps helper and SlidingWindowCompaction."""

import pytest

from minisweagent.compaction.providers.generic.sliding_window import (
    SlidingWindowCompaction,
    _group_into_steps,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _assistant(content="think", tool_calls=None):
    msg = {"role": "assistant", "content": content}
    if tool_calls:
        msg["tool_calls"] = tool_calls
    return msg


def _tool(content="output"):
    return {"role": "tool", "content": content, "tool_call_id": "x"}


def _user(content="task"):
    return {"role": "user", "content": content}


def _system(content="sys"):
    return {"role": "system", "content": content}


def _anchors():
    return [_system(), _user()]


# ---------------------------------------------------------------------------
# _group_into_steps
# ---------------------------------------------------------------------------


class TestGroupIntoSteps:
    def test_empty(self):
        assert _group_into_steps([]) == []

    def test_single_step_no_tool(self):
        msgs = [_assistant("a")]
        assert _group_into_steps(msgs) == [[_assistant("a")]]

    def test_single_step_with_tool(self):
        msgs = [_assistant("a"), _tool("o")]
        assert _group_into_steps(msgs) == [[_assistant("a"), _tool("o")]]

    def test_two_steps(self):
        msgs = [_assistant("a1"), _tool("o1"), _assistant("a2"), _tool("o2")]
        groups = _group_into_steps(msgs)
        assert len(groups) == 2
        assert groups[0] == [_assistant("a1"), _tool("o1")]
        assert groups[1] == [_assistant("a2"), _tool("o2")]

    def test_step_with_multiple_tool_messages(self):
        """Multiple tool messages belong to the same step until the next assistant."""
        msgs = [_assistant("a1"), _tool("o1a"), _tool("o1b"), _assistant("a2"), _tool("o2")]
        groups = _group_into_steps(msgs)
        assert len(groups) == 2
        assert groups[0] == [_assistant("a1"), _tool("o1a"), _tool("o1b")]
        assert groups[1] == [_assistant("a2"), _tool("o2")]

    def test_three_steps(self):
        msgs = [
            _assistant("a1"), _tool("o1"),
            _assistant("a2"), _tool("o2"),
            _assistant("a3"), _tool("o3"),
        ]
        groups = _group_into_steps(msgs)
        assert len(groups) == 3
        assert all(len(g) == 2 for g in groups)


# ---------------------------------------------------------------------------
# SlidingWindowCompaction
# ---------------------------------------------------------------------------


class TestSlidingWindowCompaction:
    def test_fewer_steps_than_window_unchanged(self):
        msgs = _anchors() + [_assistant("a1"), _tool("o1"), _assistant("a2"), _tool("o2")]
        s = SlidingWindowCompaction(keep_last_n_steps=5)
        assert s.compact(msgs) == msgs

    def test_exactly_window_size_unchanged(self):
        msgs = _anchors() + [_assistant("a1"), _tool("o1"), _assistant("a2"), _tool("o2")]
        s = SlidingWindowCompaction(keep_last_n_steps=2)
        assert s.compact(msgs) == msgs

    def test_keeps_last_n_steps(self):
        msgs = _anchors() + [
            _assistant("a1"), _tool("o1"),  # step 1 — should be dropped
            _assistant("a2"), _tool("o2"),  # step 2 — kept
            _assistant("a3"), _tool("o3"),  # step 3 — kept
        ]
        s = SlidingWindowCompaction(keep_last_n_steps=2)
        result = s.compact(msgs)
        assert result == _anchors() + [
            _assistant("a2"), _tool("o2"),
            _assistant("a3"), _tool("o3"),
        ]

    def test_always_preserves_anchors(self):
        msgs = _anchors() + [_assistant(f"a{i}") for i in range(10)]
        s = SlidingWindowCompaction(keep_last_n_steps=1)
        result = s.compact(msgs)
        assert result[0] == _system()
        assert result[1] == _user()

    def test_only_anchors_no_crash(self):
        msgs = _anchors()
        s = SlidingWindowCompaction(keep_last_n_steps=5)
        assert s.compact(msgs) == msgs

    def test_window_of_one(self):
        msgs = _anchors() + [
            _assistant("a1"), _tool("o1"),
            _assistant("a2"), _tool("o2"),
            _assistant("a3"), _tool("o3"),
        ]
        s = SlidingWindowCompaction(keep_last_n_steps=1)
        result = s.compact(msgs)
        assert result == _anchors() + [_assistant("a3"), _tool("o3")]

    def test_preserves_step_with_multiple_tools(self):
        """A step with multiple tool messages is kept or dropped as a unit."""
        msgs = _anchors() + [
            _assistant("a1"), _tool("o1a"), _tool("o1b"),  # step 1 with 2 tools
            _assistant("a2"), _tool("o2"),                  # step 2
        ]
        s = SlidingWindowCompaction(keep_last_n_steps=1)
        result = s.compact(msgs)
        # Step 1 is dropped entirely; step 2 kept
        assert result == _anchors() + [_assistant("a2"), _tool("o2")]

    def test_does_not_mutate_input(self):
        msgs = _anchors() + [_assistant("a1"), _tool("o1"), _assistant("a2"), _tool("o2")]
        original = [m.copy() for m in msgs]
        SlidingWindowCompaction(keep_last_n_steps=1).compact(msgs)
        assert msgs == original

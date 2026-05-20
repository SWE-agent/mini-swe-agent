"""Tests for get_compaction_strategy factory and NoopCompaction."""

import pytest

from minisweagent import CompactionStrategy
from minisweagent.compaction import get_compaction_strategy
from minisweagent.compaction._noop import NoopCompaction
from minisweagent.compaction.providers.generic.sliding_window import SlidingWindowCompaction


def _msgs():
    return [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "task"},
        {"role": "assistant", "content": "a1"},
        {"role": "tool", "content": "o1"},
    ]


# ---------------------------------------------------------------------------
# NoopCompaction
# ---------------------------------------------------------------------------


class TestNoopCompaction:
    def test_returns_same_list(self):
        msgs = _msgs()
        result = NoopCompaction().compact(msgs)
        assert result == msgs

    def test_returns_same_object(self):
        """Noop should return the exact same list, not a copy."""
        msgs = _msgs()
        assert NoopCompaction().compact(msgs) is msgs


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


class TestGetCompactionStrategy:
    def test_none_returns_noop(self):
        s = get_compaction_strategy("none")
        assert isinstance(s, NoopCompaction)

    def test_default_is_none(self):
        s = get_compaction_strategy()
        assert isinstance(s, NoopCompaction)

    def test_sliding_window(self):
        s = get_compaction_strategy("generic/sliding_window")
        assert isinstance(s, SlidingWindowCompaction)

    def test_sliding_window_kwargs_forwarded(self):
        s = get_compaction_strategy("generic/sliding_window", keep_last_n_steps=3)
        assert isinstance(s, SlidingWindowCompaction)
        assert s.keep_last_n_steps == 3

    def test_unknown_strategy_raises(self):
        with pytest.raises(ValueError, match="Unknown compaction strategy"):
            get_compaction_strategy("does/not/exist")

    def test_full_import_path_works(self):
        """Users can pass a dotted import path to a custom class."""
        s = get_compaction_strategy(
            "minisweagent.compaction.providers.generic.sliding_window.SlidingWindowCompaction",
            keep_last_n_steps=7,
        )
        assert isinstance(s, SlidingWindowCompaction)
        assert s.keep_last_n_steps == 7


# ---------------------------------------------------------------------------
# Protocol compliance
# ---------------------------------------------------------------------------


class TestCompactionStrategyProtocol:
    @pytest.mark.parametrize(
        "name",
        ["none", "generic/sliding_window"],
    )
    def test_isinstance_check(self, name):
        s = get_compaction_strategy(name)
        assert isinstance(s, CompactionStrategy)

    def test_custom_class_satisfies_protocol(self):
        class MyCompaction:
            def compact(self, messages):
                return messages

        assert isinstance(MyCompaction(), CompactionStrategy)

    def test_class_missing_compact_fails_protocol(self):
        class Bad:
            pass

        assert not isinstance(Bad(), CompactionStrategy)

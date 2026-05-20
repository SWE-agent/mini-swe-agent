"""Compaction strategy factory.

Usage::

    from minisweagent.compaction import get_compaction_strategy

    strategy = get_compaction_strategy("generic/sliding_window", keep_last_n_steps=8)
    compacted = strategy.compact(messages)
"""

import importlib
from typing import Any

from minisweagent import CompactionStrategy

_STRATEGY_REGISTRY: dict[str, str] = {
    "none": "minisweagent.compaction._noop.NoopCompaction",
    "generic/sliding_window": "minisweagent.compaction.providers.generic.sliding_window.SlidingWindowCompaction",
    "generic/summarize": "minisweagent.compaction.providers.generic.summarize.SummarizeCompaction",
    "anthropic/summarize": "minisweagent.compaction.providers.anthropic.summarize.AnthropicSummarizeCompaction",
}


def get_compaction_strategy(strategy: str = "none", **kwargs: Any) -> CompactionStrategy:
    """Return an initialized compaction strategy by name.

    Args:
        strategy: One of the keys in the strategy registry, e.g.
            ``"none"``, ``"generic/sliding_window"``, ``"anthropic/summarize"``.
            You may also pass a full dotted import path to a custom class.
        **kwargs: Forwarded to the strategy constructor.
    """
    full_path = _STRATEGY_REGISTRY.get(strategy, strategy)
    try:
        module_name, class_name = full_path.rsplit(".", 1)
        module = importlib.import_module(module_name)
        cls = getattr(module, class_name)
    except (ValueError, ImportError, AttributeError) as e:
        available = list(_STRATEGY_REGISTRY)
        msg = f"Unknown compaction strategy: {strategy!r} (tried {full_path!r}). Available: {available}"
        raise ValueError(msg) from e
    return cls(**kwargs)


__all__ = ["get_compaction_strategy"]

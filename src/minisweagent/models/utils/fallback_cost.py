"""Fallback pricing for models that litellm cannot price, using the optional `llm-prices` package."""

from typing import Any


def _tokens(obj: Any, name: str) -> int:
    return value if isinstance(value := getattr(obj, name, None), int) else 0


def fallback_cost(model_name: str, usage: Any) -> float | None:
    """Cost of one response from the `llm-prices` table (`pip install 'mini-swe-agent[prices]'`).

    Returns None if `llm-prices` is not installed, the model is not in the table, or the cost is not > 0.
    """
    try:
        import llm_prices
    except ImportError:
        return None
    if (price := llm_prices.lookup(model_name)) is None:
        return None
    cached = _tokens(getattr(usage, "prompt_tokens_details", None), "cached_tokens")
    written = _tokens(usage, "cache_creation_input_tokens")
    breakdown = price.cost_usage(
        llm_prices.Usage(
            input=max(_tokens(usage, "prompt_tokens") - cached - written, 0),
            output=_tokens(usage, "completion_tokens"),
            cache_read=cached,
            cache_write_5m=written,
        )
    )
    return breakdown.total if breakdown and breakdown.total > 0 else None

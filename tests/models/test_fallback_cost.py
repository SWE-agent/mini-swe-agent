import pytest
from litellm import ModelResponse

from minisweagent.models.litellm_model import LitellmModel
from minisweagent.models.utils.fallback_cost import fallback_cost

llm_prices = pytest.importorskip("llm_prices")


def _response(model_name: str, **usage) -> ModelResponse:
    return ModelResponse(model=model_name, choices=[{"message": {"role": "assistant", "content": "hi"}}], usage=usage)


@pytest.mark.parametrize(
    ("model_name", "usage", "expected"),
    [
        (
            "openrouter/moonshotai/kimi-k2-0905",
            {"prompt_tokens": 274_812, "completion_tokens": 156, "total_tokens": 274_968},
            llm_prices.Usage(input=274_812, output=156),
        ),
        (
            "openrouter/z-ai/glm-5.3",
            {
                "prompt_tokens": 10_000,
                "completion_tokens": 500,
                "total_tokens": 10_500,
                "prompt_tokens_details": {"cached_tokens": 8_000},
            },
            llm_prices.Usage(input=2_000, cache_read=8_000, output=500),
        ),
    ],
)
def test_unpriced_litellm_model_falls_back_to_llm_prices(model_name, usage, expected):
    """litellm has no price for these OpenRouter ids, so without the fallback _calculate_cost raises."""
    cost = LitellmModel(model_name=model_name, cost_tracking="default")._calculate_cost(_response(model_name, **usage))
    assert cost == {"cost": llm_prices.lookup(model_name).cost_usage(expected).total}
    assert cost["cost"] > 0


def test_unknown_model_still_raises():
    model_name = "openai/not-a-real-model-xyz"
    response = _response(model_name, prompt_tokens=10, completion_tokens=5, total_tokens=15)
    assert fallback_cost(model_name, response.usage) is None
    with pytest.raises(RuntimeError, match="Error calculating cost"):
        LitellmModel(model_name=model_name, cost_tracking="default")._calculate_cost(response)
    assert LitellmModel(model_name=model_name, cost_tracking="ignore_errors")._calculate_cost(response) == {"cost": 0.0}


def test_anthropic_cache_writes_are_not_priced_as_input():
    usage = _response("x", prompt_tokens=3_000, completion_tokens=100, total_tokens=3_100).usage
    usage.cache_creation_input_tokens = 2_000
    assert fallback_cost("claude-sonnet-4-5-20250929", usage) == pytest.approx(
        llm_prices.lookup("claude-sonnet-4-5-20250929")
        .cost_usage(llm_prices.Usage(input=1_000, cache_write_5m=2_000, output=100))
        .total
    )

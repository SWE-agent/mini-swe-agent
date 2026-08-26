import pytest

import minisweagent.models
from minisweagent.exceptions import ContextWindowExceeded, is_context_window_error
from minisweagent.models.extra.roulette import RouletteModel
from minisweagent.models.litellm_model import LitellmModel, LitellmModelConfig
from minisweagent.models.openrouter_model import OpenRouterModel, OpenRouterModelConfig
from minisweagent.models.openrouter_response_model import OpenRouterResponseModel, OpenRouterResponseModelConfig
from minisweagent.models.test_models import DeterministicModel
from minisweagent.models.utils.text_generation import request_kwargs, responses_kwargs


class _Message:
    content = [{"type": "text", "text": "summary text"}]

    def model_dump(self):
        return {"role": "assistant", "content": self.content}


class _Choice:
    message = _Message()


class _ChatResponse:
    choices = [_Choice()]

    def model_dump(self, **kwargs):
        return {"choices": [{"message": self.choices[0].message.model_dump()}]}


class _FakeLitellmModel(LitellmModel):
    abort_exceptions = [ContextWindowExceeded]

    def __init__(self):
        self.config = LitellmModelConfig(model_name="test")
        self.use_tools = None

    def _query(self, messages, *, use_tools=True, **kwargs):
        self.use_tools = use_tools
        return _ChatResponse()

    def _calculate_cost(self, response):
        return {"cost": 0.25}

    def _parse_actions(self, response):
        raise AssertionError("summary generation must bypass action parsing")


class _FakeOpenRouterModel(OpenRouterModel):
    def __init__(self):
        self.config = OpenRouterModelConfig(model_name="test")
        self.use_tools = None

    def _query(self, messages, *, use_tools=True, **kwargs):
        self.use_tools = use_tools
        return {
            "choices": [{"message": {"role": "assistant", "content": "router summary"}}],
            "usage": {"cost": 0.5},
        }

    def _parse_actions(self, response):
        raise AssertionError("summary generation must bypass action parsing")


class _FakeOpenRouterResponseModel(OpenRouterResponseModel):
    def __init__(self):
        self.config = OpenRouterResponseModelConfig(model_name="test")
        self.use_tools = None
        self.kwargs = {}

    def _query(self, messages, *, use_tools=True, **kwargs):
        self.use_tools = use_tools
        self.kwargs = kwargs
        return {
            "output": [
                {
                    "type": "message",
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": "responses summary"}],
                }
            ],
            "usage": {"cost": 0.75},
        }

    def _parse_actions(self, response):
        raise AssertionError("summary generation must bypass action parsing")


@pytest.mark.parametrize(
    ("model", "text", "cost"),
    [
        (_FakeLitellmModel(), "summary text", 0.25),
        (_FakeOpenRouterModel(), "router summary", 0.5),
        (_FakeOpenRouterResponseModel(), "responses summary", 0.75),
    ],
)
def test_generate_text_is_tool_free_normalized_and_accounted(model, text, cost, reset_global_stats):
    result = model.generate_text([{"role": "user", "content": "summarize"}])
    assert result["text"] == text
    assert result["cost"] == cost
    assert result["response"]
    assert model.use_tools is False
    assert minisweagent.models.GLOBAL_MODEL_STATS.cost == cost
    assert minisweagent.models.GLOBAL_MODEL_STATS.n_calls == 1


def test_deterministic_summary_outputs_are_independent(reset_global_stats):
    model = DeterministicModel(
        outputs=[{"role": "assistant", "content": "action", "extra": {"actions": []}}],
        summary_outputs=["first summary", "second summary"],
        cost_per_call=0.4,
    )
    assert model.generate_text([])["text"] == "first summary"
    assert model.query([])["content"] == "action"
    assert model.generate_text([])["text"] == "second summary"
    assert minisweagent.models.GLOBAL_MODEL_STATS.cost == pytest.approx(1.2)
    assert minisweagent.models.GLOBAL_MODEL_STATS.n_calls == 3


def test_roulette_delegates_summary_and_message_formatting(reset_global_stats):
    model = RouletteModel(
        model_kwargs=[
            {
                "model_class": "deterministic",
                "model_name": "deterministic",
                "outputs": [],
                "summary_outputs": ["roulette checkpoint"],
            }
        ]
    )

    assert model.format_message(role="user", content="hello")["content"] == "hello"
    assert model.generate_text([])["text"] == "roulette checkpoint"


def test_responses_generation_normalizes_output_limit(reset_global_stats):
    model = _FakeOpenRouterResponseModel()

    model.generate_text([], max_tokens=123)

    assert model.kwargs["max_output_tokens"] == 123
    assert "max_tokens" not in model.kwargs


def test_context_overflow_detection():
    for error in [
        {"error": {"code": "context_length_exceeded", "message": "request too large"}},
        ValueError("maximum context window limit exceeded"),
        RuntimeError("input token limit is too long"),
    ]:
        assert is_context_window_error(error)


def test_request_kwargs_remove_tools_for_text_generation():
    assert request_kwargs({"tools": ["configured"], "temperature": 0.2}, {"tools": ["caller"]}, None) == {
        "temperature": 0.2
    }
    assert request_kwargs({"tools": ["configured"]}, {"tools": ["caller"]}, [{"name": "bash"}]) == {
        "tools": [{"name": "bash"}]
    }


def test_responses_kwargs_normalize_output_limit():
    assert responses_kwargs({"max_tokens": 123}) == {"max_output_tokens": 123}
    assert responses_kwargs({"max_tokens": 123, "max_output_tokens": 456}) == {"max_output_tokens": 456}

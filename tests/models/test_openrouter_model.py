from unittest.mock import patch

import pytest
from tenacity import Retrying, stop_after_attempt, wait_none

from minisweagent.models.openrouter_model import OpenRouterAPIError, OpenRouterModel


@pytest.mark.parametrize(
    ("response",),  # noqa: PT006
    [
        ({"error": {"message": "Provider returned an error"}},),
        ({"choices": []},),
        ({"choices": [{}]},),
        ({"choices": "invalid"},),
        ([],),
    ],
)
def test_openrouter_model_rejects_invalid_chat_completion(response) -> None:
    model = OpenRouterModel(model_name="test/model")

    with pytest.raises(OpenRouterAPIError, match="Invalid chat completion response"):
        model._validate_response(response)


def test_openrouter_model_retries_invalid_chat_completion() -> None:
    model = OpenRouterModel(model_name="test/model")
    valid_response = {
        "choices": [
            {
                "message": {
                    "tool_calls": [
                        {"id": "call_1", "function": {"name": "bash", "arguments": '{"command": "echo ok"}'}}
                    ]
                },
                "finish_reason": "tool_calls",
            }
        ],
        "usage": {"cost": 0.1},
    }

    with (
        patch.object(model, "_query", side_effect=[{"error": {"message": "temporary error"}}, valid_response]) as query,
        patch(
            "minisweagent.models.openrouter_model.retry",
            return_value=Retrying(stop=stop_after_attempt(2), wait=wait_none(), reraise=True),
        ),
    ):
        assert model.query([{"role": "user", "content": "hello"}])["extra"]["response"] == valid_response

    assert query.call_count == 2

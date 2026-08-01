import json
import os
from unittest.mock import Mock, patch

import pytest
import requests

from minisweagent.models import GLOBAL_MODEL_STATS
from minisweagent.models.orcarouter_model import OrcaRouterAuthenticationError, OrcaRouterModel


@pytest.fixture
def mock_response():
    """Mock successful OrcaRouter API response (shape taken from a real API call)."""
    return {
        "id": "gen-1785600138-dqBDVf7FnAj7wD1og5ZF",
        "model": "openai/gpt-5.6-terra",
        "provider": "OpenAI",
        "choices": [
            {
                "finish_reason": "tool_calls",
                "message": {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "type": "function",
                            "function": {"name": "bash", "arguments": '{"command": "echo hello"}'},
                        }
                    ],
                },
            }
        ],
        "usage": {"prompt_tokens": 16, "completion_tokens": 13, "total_tokens": 29, "cost": 0.000157},
    }


@pytest.fixture
def mock_response_no_cost(mock_response):
    return mock_response | {"usage": {"prompt_tokens": 16, "completion_tokens": 13, "total_tokens": 29}}


def test_orcarouter_model_successful_query(mock_response):
    with patch.dict(os.environ, {"ORCAROUTER_API_KEY": "sk-orca-test-key"}):
        model = OrcaRouterModel(model_name="anthropic/claude-sonnet-4.6", model_kwargs={"temperature": 0.7})
        initial_cost = GLOBAL_MODEL_STATS.cost

        with patch("requests.post") as mock_post:
            mock_post.return_value.status_code = 200
            mock_post.return_value.json.return_value = mock_response
            mock_post.return_value.raise_for_status.return_value = None

            result = model.query([{"role": "user", "content": "Say hello from bash"}])

        assert mock_post.call_args[0][0] == "https://api.orcarouter.ai/v1/chat/completions"
        assert mock_post.call_args[1]["headers"]["Authorization"] == "Bearer sk-orca-test-key"
        assert mock_post.call_args[1]["headers"]["X-Title"] == "mini-swe-agent"

        payload = json.loads(mock_post.call_args[1]["data"])
        assert payload["model"] == "anthropic/claude-sonnet-4.6"
        assert payload["messages"] == [{"role": "user", "content": "Say hello from bash"}]
        assert payload["temperature"] == 0.7
        assert payload["usage"]["include"] is True
        assert [tool["function"]["name"] for tool in payload["tools"]] == ["bash"]

        assert result["extra"]["actions"] == [{"command": "echo hello", "tool_call_id": "call_1"}]
        assert result["extra"]["response"] == mock_response
        assert GLOBAL_MODEL_STATS.cost == initial_cost + 0.000157


def test_orcarouter_model_authentication_error():
    """A 401 must abort the run rather than be retried, so it maps to an abort exception."""
    with patch.dict(os.environ, {"ORCAROUTER_API_KEY": "sk-orca-invalid"}):
        model = OrcaRouterModel(model_name="anthropic/claude-sonnet-4.6")

        with patch("requests.post") as mock_post:
            mock_post.return_value = Mock(status_code=401, text="Unauthorized")
            mock_post.return_value.raise_for_status.side_effect = requests.exceptions.HTTPError()

            with pytest.raises(OrcaRouterAuthenticationError) as exc_info:
                model.query([{"role": "user", "content": "test"}])

        assert "mini-extra config set ORCAROUTER_API_KEY" in str(exc_info.value)
        assert mock_post.call_count == 1
        assert OrcaRouterAuthenticationError in model.abort_exceptions


@pytest.mark.parametrize(
    ("cost_tracking", "expect_error"),
    [("default", True), ("ignore_errors", False)],
)
def test_orcarouter_model_missing_cost(mock_response_no_cost, cost_tracking, expect_error):
    with patch.dict(os.environ, {"ORCAROUTER_API_KEY": "sk-orca-test-key"}):
        model = OrcaRouterModel(model_name="anthropic/claude-sonnet-4.6", cost_tracking=cost_tracking)
        initial_cost = GLOBAL_MODEL_STATS.cost

        with patch("requests.post") as mock_post:
            mock_post.return_value.status_code = 200
            mock_post.return_value.json.return_value = mock_response_no_cost
            mock_post.return_value.raise_for_status.return_value = None

            if expect_error:
                with pytest.raises(RuntimeError, match="No valid cost information available"):
                    model.query([{"role": "user", "content": "test"}])
            else:
                assert model.query([{"role": "user", "content": "test"}])["extra"]["cost"] == 0.0

        assert GLOBAL_MODEL_STATS.cost == initial_cost

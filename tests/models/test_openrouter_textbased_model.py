import json
import os
from pathlib import Path
from unittest.mock import Mock, patch

import pytest
import requests
import yaml

from minisweagent.models import GLOBAL_MODEL_STATS, get_model
from minisweagent.models.openrouter_model import (
    OpenRouterAuthenticationError,
)
from minisweagent.models.openrouter_textbased_model import (
    OpenRouterTextbasedModel,
)


@pytest.fixture
def mock_response():
    """Create a mock successful OpenRouter API response."""
    # Response must include bash block to avoid FormatError from parse_action
    return {
        "choices": [{"message": {"content": "```mswea_bash_command\necho '2+2 equals 4'\n```"}}],
        "usage": {
            "prompt_tokens": 16,
            "completion_tokens": 13,
            "total_tokens": 29,
            "cost": 0.000243,
            "is_byok": False,
            "prompt_tokens_details": {"cached_tokens": 0, "audio_tokens": 0},
            "cost_details": {
                "upstream_inference_cost": None,
                "upstream_inference_prompt_cost": 4.8e-05,
                "upstream_inference_completions_cost": 0.000195,
            },
        },
    }


@pytest.fixture
def mock_response_no_cost():
    """Create a mock OpenRouter API response without cost information."""
    # Response must include bash block to avoid FormatError from parse_action
    return {
        "choices": [{"message": {"content": "```mswea_bash_command\necho '2+2 equals 4'\n```"}}],
        "usage": {"prompt_tokens": 16, "completion_tokens": 13, "total_tokens": 29},
    }


def test_openrouter_model_successful_query(mock_response):
    """Test successful OpenRouter API query with cost tracking."""
    with patch.dict(os.environ, {"OPENROUTER_API_KEY": "test-key"}):
        model = OpenRouterTextbasedModel(model_name="anthropic/claude-3.5-sonnet", model_kwargs={"temperature": 0.7})

        initial_cost = GLOBAL_MODEL_STATS.cost

        with patch("requests.post") as mock_post:
            # Mock successful response
            mock_post.return_value.status_code = 200
            mock_post.return_value.json.return_value = mock_response
            mock_post.return_value.raise_for_status.return_value = None

            messages = [{"role": "user", "content": "Hello! What is 2+2?"}]
            result = model.query(messages)

            # Verify the request was made correctly
            mock_post.assert_called_once()
            call_args = mock_post.call_args

            # Check URL (first positional argument)
            assert call_args[0][0] == "https://openrouter.ai/api/v1/chat/completions"

            # Check headers
            headers = call_args[1]["headers"]
            assert headers["Authorization"] == "Bearer test-key"
            assert headers["Content-Type"] == "application/json"

            # Check payload
            payload = json.loads(call_args[1]["data"])
            assert payload["model"] == "anthropic/claude-3.5-sonnet"
            assert payload["messages"] == messages
            assert payload["usage"]["include"] is True
            assert payload["temperature"] == 0.7

            # Verify response
            assert result["content"] == "```mswea_bash_command\necho '2+2 equals 4'\n```"
            assert result["extra"]["actions"] == [{"command": "echo '2+2 equals 4'"}]
            assert result["extra"]["response"] == mock_response

            # Verify cost tracking
            assert GLOBAL_MODEL_STATS.cost == initial_cost + 0.000243


def test_minimax_flash_payload_preserves_model_options_and_multimodal_messages(mock_response):
    """Verify the configured OpenRouter Flash route reaches the HTTP payload unchanged."""
    with patch.dict(os.environ, {"OPENROUTER_API_KEY": "test-key"}):
        model = OpenRouterTextbasedModel(
            model_name="minimax/minimax-m3.1-flash:free",
            model_kwargs={"reasoning_effort": "max", "temperature": 0.0},
        )

        with patch("requests.post") as mock_post:
            mock_post.return_value.status_code = 200
            mock_post.return_value.json.return_value = mock_response
            mock_post.return_value.raise_for_status.return_value = None

            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Inspect this image."},
                        {"type": "image_url", "image_url": {"url": "https://example.test/image.png"}},
                    ],
                }
            ]
            model.query(messages)

            payload = json.loads(mock_post.call_args.kwargs["data"])
            assert payload["model"] == "minimax/minimax-m3.1-flash:free"
            assert payload["reasoning_effort"] == "max"
            assert payload["temperature"] == 0.0
            assert payload["messages"] == messages


@pytest.mark.parametrize(
    "model_name",
    [None, "minimax/minimax-m3", "minimax/minimax-m2", "custom/model"],
)
def test_minimax_benchmark_http_payload(model_name, monkeypatch):
    config = yaml.safe_load(
        (Path(__file__).parents[2] / "src/minisweagent/config/benchmarks/swebench_xml.yaml").read_text()
    )["model"]
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    model = get_model(model_name, config)
    messages = [{"role": "user", "content": "Run a command."}]
    response = requests.Response()
    response.status_code = 200
    response._content = json.dumps(
        {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": "",
                        "tool_calls": [
                            {
                                "id": "call_1",
                                "type": "function",
                                "function": {"name": "bash", "arguments": '{"command":"pwd"}'},
                            }
                        ],
                    }
                }
            ],
            "usage": {"cost": 0},
        }
    ).encode()
    with patch("requests.sessions.Session.send", return_value=response) as send:
        assert model.query(messages)["extra"]["actions"] == [{"command": "pwd", "tool_call_id": "call_1"}]
        request = send.call_args.args[0]
        payload = json.loads(request.body)
        assert request.url == "https://openrouter.ai/api/v1/chat/completions"
        assert payload["model"] == (model_name or "minimax/minimax-m3.1-flash:free")
        assert payload["messages"] == messages
        assert payload["tools"][0]["function"]["name"] == "bash"
        assert "service_tier" not in payload
        assert "thinking" not in payload
    saved = model.serialize()["info"]["config"]["model"]
    assert get_model(config=saved | {"model_class": config["model_class"]}).config.model_name == payload["model"]


def test_openrouter_model_authentication_error():
    """Test authentication error handling."""
    with patch.dict(os.environ, {"OPENROUTER_API_KEY": "invalid-key"}):
        model = OpenRouterTextbasedModel(model_name="anthropic/claude-3.5-sonnet")

        with patch("requests.post") as mock_post:
            # Mock 401 authentication error
            mock_response = Mock()
            mock_response.status_code = 401
            mock_response.text = "Unauthorized"
            mock_post.return_value = mock_response
            mock_post.return_value.raise_for_status.side_effect = requests.exceptions.HTTPError()

            messages = [{"role": "user", "content": "test"}]

            with pytest.raises(OpenRouterAuthenticationError) as exc_info:
                model._query(messages)

            assert "Authentication failed" in str(exc_info.value)
            assert "mini-extra config set OPENROUTER_API_KEY" in str(exc_info.value)


def test_openrouter_model_no_cost_information(mock_response_no_cost):
    """Test error when cost information is missing."""
    with patch.dict(os.environ, {"OPENROUTER_API_KEY": "test-key"}):
        model = OpenRouterTextbasedModel(model_name="anthropic/claude-3.5-sonnet")

        with patch("requests.post") as mock_post:
            mock_post.return_value.status_code = 200
            mock_post.return_value.json.return_value = mock_response_no_cost
            mock_post.return_value.raise_for_status.return_value = None

            messages = [{"role": "user", "content": "test"}]

            with pytest.raises(RuntimeError) as exc_info:
                model.query(messages)

            assert "No valid cost information available" in str(exc_info.value)
            assert "MSWEA_COST_TRACKING='ignore_errors'" in str(exc_info.value)


def test_openrouter_model_free_model_zero_cost(mock_response_no_cost):
    """Test that free models with zero cost work correctly when cost_tracking='ignore_errors' is set."""
    with patch.dict(os.environ, {"OPENROUTER_API_KEY": "test-key"}):
        model = OpenRouterTextbasedModel(model_name="anthropic/claude-3.5-sonnet", cost_tracking="ignore_errors")

        initial_cost = GLOBAL_MODEL_STATS.cost

        with patch("requests.post") as mock_post:
            mock_post.return_value.status_code = 200
            mock_post.return_value.json.return_value = mock_response_no_cost
            mock_post.return_value.raise_for_status.return_value = None

            messages = [{"role": "user", "content": "test"}]

            # With cost_tracking='ignore_errors', free models should work without raising an error
            result = model.query(messages)

            # Verify response
            assert result["content"] == "```mswea_bash_command\necho '2+2 equals 4'\n```"
            assert result["extra"]["actions"] == [{"command": "echo '2+2 equals 4'"}]
            assert result["extra"]["response"] == mock_response_no_cost

            # Verify cost tracking with zero cost (not added to global stats when zero)
            # Cost should not be added to global stats since it's zero
            assert GLOBAL_MODEL_STATS.cost == initial_cost


def test_openrouter_model_config():
    """Test OpenRouter model configuration."""
    with patch.dict(os.environ, {"OPENROUTER_API_KEY": "test-key"}):
        model = OpenRouterTextbasedModel(
            model_name="anthropic/claude-3.5-sonnet", model_kwargs={"temperature": 0.5, "max_tokens": 1000}
        )

        assert model.config.model_name == "anthropic/claude-3.5-sonnet"
        assert model.config.model_kwargs == {"temperature": 0.5, "max_tokens": 1000}
        assert model._api_key == "test-key"


def test_openrouter_model_get_template_vars():
    """Test get_template_vars method."""
    with patch.dict(os.environ, {"OPENROUTER_API_KEY": "test-key"}):
        model = OpenRouterTextbasedModel(model_name="anthropic/claude-3.5-sonnet", model_kwargs={"temperature": 0.7})

        template_vars = model.get_template_vars()

        assert template_vars["model_name"] == "anthropic/claude-3.5-sonnet"
        assert template_vars["model_kwargs"] == {"temperature": 0.7}


def test_openrouter_model_no_api_key():
    """Test behavior when no API key is provided."""
    with patch.dict(os.environ, {}, clear=True):
        model = OpenRouterTextbasedModel(model_name="anthropic/claude-3.5-sonnet")

        assert model._api_key == ""

        with patch("requests.post") as mock_post:
            mock_response = Mock()
            mock_response.status_code = 401
            mock_response.text = "Unauthorized"
            mock_post.return_value = mock_response
            mock_post.return_value.raise_for_status.side_effect = requests.exceptions.HTTPError()

            messages = [{"role": "user", "content": "test"}]

            with pytest.raises(OpenRouterAuthenticationError):
                model._query(messages)

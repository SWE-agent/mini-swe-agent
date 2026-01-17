import json
import os
from unittest.mock import Mock, patch

import pytest
import requests

from minisweagent.models import GLOBAL_MODEL_STATS
from minisweagent.models.openrouter_model import (
    OpenRouterAuthenticationError,
    OpenRouterModel,
)


@pytest.fixture
def mock_response():
    """Create a mock successful OpenRouter API response."""
    return {
        "choices": [{"message": {"content": "Hello! 2+2 equals 4."}}],
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
    return {
        "choices": [{"message": {"content": "Hello! 2+2 equals 4."}}],
        "usage": {"prompt_tokens": 16, "completion_tokens": 13, "total_tokens": 29},
    }


def test_openrouter_model_successful_query(mock_response):
    """Test successful OpenRouter API query with cost tracking."""
    with patch.dict(os.environ, {"OPENROUTER_API_KEY": "test-key"}):
        model = OpenRouterModel(model_name="anthropic/claude-3.5-sonnet", model_kwargs={"temperature": 0.7})

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
            assert result["content"] == "Hello! 2+2 equals 4."
            assert result["extra"]["response"] == mock_response

            # Verify cost tracking
            assert model.cost == 0.000243
            assert model.n_calls == 1
            assert GLOBAL_MODEL_STATS.cost == initial_cost + 0.000243


def test_openrouter_model_authentication_error():
    """Test authentication error handling."""
    with patch.dict(os.environ, {"OPENROUTER_API_KEY": "invalid-key"}):
        model = OpenRouterModel(model_name="anthropic/claude-3.5-sonnet")

        with patch("requests.post") as mock_post:
            # Mock 401 authentication error
            mock_response = Mock()
            mock_response.status_code = 401
            mock_response.text = "Unauthorized"
            mock_post.return_value = mock_response
            mock_post.return_value.raise_for_status.side_effect = requests.exceptions.HTTPError()

            messages = [{"role": "user", "content": "test"}]

            # Patch the retry decorator to avoid waiting (auth errors don't retry anyway)
            with patch("minisweagent.models.openrouter_model.retry", lambda **kwargs: lambda f: f):
                with pytest.raises(OpenRouterAuthenticationError) as exc_info:
                    model._query(messages)

                assert "Authentication failed" in str(exc_info.value)
                assert "mini-extra config set OPENROUTER_API_KEY" in str(exc_info.value)


def test_openrouter_model_no_cost_information(mock_response_no_cost):
    """Test error when cost information is missing."""
    with patch.dict(os.environ, {"OPENROUTER_API_KEY": "test-key"}):
        model = OpenRouterModel(model_name="anthropic/claude-3.5-sonnet")

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
        model = OpenRouterModel(model_name="anthropic/claude-3.5-sonnet", cost_tracking="ignore_errors")

        initial_cost = GLOBAL_MODEL_STATS.cost

        with patch("requests.post") as mock_post:
            mock_post.return_value.status_code = 200
            mock_post.return_value.json.return_value = mock_response_no_cost
            mock_post.return_value.raise_for_status.return_value = None

            messages = [{"role": "user", "content": "test"}]

            # With cost_tracking='ignore_errors', free models should work without raising an error
            result = model.query(messages)

            # Verify response
            assert result["content"] == "Hello! 2+2 equals 4."
            assert result["extra"]["response"] == mock_response_no_cost

            # Verify cost tracking with zero cost (not added to global stats when zero)
            assert model.cost == 0.0
            assert model.n_calls == 1
            # Cost should not be added to global stats since it's zero
            assert GLOBAL_MODEL_STATS.cost == initial_cost


def test_openrouter_model_config():
    """Test OpenRouter model configuration."""
    with patch.dict(os.environ, {"OPENROUTER_API_KEY": "test-key"}):
        model = OpenRouterModel(
            model_name="anthropic/claude-3.5-sonnet", model_kwargs={"temperature": 0.5, "max_tokens": 1000}
        )

        assert model.config.model_name == "anthropic/claude-3.5-sonnet"
        assert model.config.model_kwargs == {"temperature": 0.5, "max_tokens": 1000}
        assert model._api_key == "test-key"
        assert model.cost == 0.0
        assert model.n_calls == 0


def test_openrouter_model_get_template_vars():
    """Test get_template_vars method."""
    with patch.dict(os.environ, {"OPENROUTER_API_KEY": "test-key"}):
        model = OpenRouterModel(model_name="anthropic/claude-3.5-sonnet", model_kwargs={"temperature": 0.7})

        # Simulate some usage
        model.cost = 0.001234
        model.n_calls = 5

        template_vars = model.get_template_vars()

        assert template_vars["model_name"] == "anthropic/claude-3.5-sonnet"
        assert template_vars["model_kwargs"] == {"temperature": 0.7}
        assert template_vars["n_model_calls"] == 5
        assert template_vars["model_cost"] == 0.001234


def test_openrouter_model_no_api_key():
    """Test behavior when no API key is provided."""
    with patch.dict(os.environ, {}, clear=True):
        model = OpenRouterModel(model_name="anthropic/claude-3.5-sonnet")

        assert model._api_key == ""

        with patch("requests.post") as mock_post:
            mock_response = Mock()
            mock_response.status_code = 401
            mock_response.text = "Unauthorized"
            mock_post.return_value = mock_response
            mock_post.return_value.raise_for_status.side_effect = requests.exceptions.HTTPError()

            messages = [{"role": "user", "content": "test"}]

            # Patch the retry decorator to avoid waiting
            with patch("minisweagent.models.openrouter_model.retry", lambda **kwargs: lambda f: f):
                with pytest.raises(OpenRouterAuthenticationError):
                    model._query(messages)

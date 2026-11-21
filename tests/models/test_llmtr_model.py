import json
import os
from unittest.mock import Mock, patch

import pytest
import requests

from minisweagent.models import GLOBAL_MODEL_STATS, get_model_class
from minisweagent.models.llmtr_model import (
    LLMTRAuthenticationError,
    LLMTRModel,
)


def test_llmtr_model_class_shortcut():
    assert get_model_class("any-model", "llmtr") is LLMTRModel


@pytest.fixture
def mock_response():
    return {
        "choices": [
            {
                "message": {
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "type": "function",
                            "function": {"name": "bash", "arguments": '{"command": "echo ok"}'},
                        }
                    ],
                },
                "finish_reason": "tool_calls",
            }
        ],
        "usage": {"prompt_tokens": 16, "completion_tokens": 13, "total_tokens": 29, "cost": 0.000243},
    }


@pytest.fixture
def mock_response_no_cost():
    return {
        "choices": [
            {
                "message": {
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "type": "function",
                            "function": {"name": "bash", "arguments": '{"command": "echo ok"}'},
                        }
                    ],
                },
                "finish_reason": "tool_calls",
            }
        ],
        "usage": {"prompt_tokens": 16, "completion_tokens": 13, "total_tokens": 29},
    }


def test_llmtr_model_successful_query(mock_response):
    with patch.dict(os.environ, {"LLMTR_API_KEY": "test-llmtr-key"}):
        model = LLMTRModel(model_name="minimax/minimax-m3.1-flash", model_kwargs={"max_tokens": 64})

        initial_cost = GLOBAL_MODEL_STATS.cost

        with patch("requests.post") as mock_post:
            mock_post.return_value.status_code = 200
            mock_post.return_value.json.return_value = mock_response
            mock_post.return_value.raise_for_status.return_value = None

            messages = [{"role": "user", "content": "Say ok in one word."}]
            result = model.query(messages)

            mock_post.assert_called_once()
            call_args = mock_post.call_args
            assert call_args[0][0] == "https://llmtr.com/v1/chat/completions"
            headers = call_args[1]["headers"]
            assert headers["Authorization"] == "Bearer test-llmtr-key"
            assert headers["Content-Type"] == "application/json"
            payload = json.loads(call_args[1]["data"])
            assert payload["model"] == "minimax/minimax-m3.1-flash"
            assert payload["messages"] == messages
            assert payload["max_tokens"] == 64
            assert "tools" in payload

            assert result["extra"]["actions"] == [{"command": "echo ok", "tool_call_id": "call_1"}]
            assert result["extra"]["response"] == mock_response
            assert GLOBAL_MODEL_STATS.cost == initial_cost + 0.000243


def test_llmtr_model_authentication_error():
    with patch.dict(os.environ, {"LLMTR_API_KEY": "invalid-key"}):
        model = LLMTRModel(model_name="minimax/minimax-m3.1-flash")

        with patch("requests.post") as mock_post:
            mock_response = Mock()
            mock_response.status_code = 401
            mock_response.text = "Unauthorized"
            mock_post.return_value = mock_response
            mock_post.return_value.raise_for_status.side_effect = requests.exceptions.HTTPError()

            with pytest.raises(LLMTRAuthenticationError) as exc_info:
                model._query([{"role": "user", "content": "test"}])

            assert "Authentication failed" in str(exc_info.value)
            assert "mini-extra config set LLMTR_API_KEY" in str(exc_info.value)


def test_llmtr_model_no_cost_information(mock_response_no_cost):
    with patch.dict(os.environ, {"LLMTR_API_KEY": "test-llmtr-key"}):
        model = LLMTRModel(model_name="minimax/minimax-m3.1-flash")

        with patch("requests.post") as mock_post:
            mock_post.return_value.status_code = 200
            mock_post.return_value.json.return_value = mock_response_no_cost
            mock_post.return_value.raise_for_status.return_value = None

            with pytest.raises(RuntimeError) as exc_info:
                model.query([{"role": "user", "content": "test"}])

            assert "No valid cost information available" in str(exc_info.value)
            assert "MSWEA_COST_TRACKING='ignore_errors'" in str(exc_info.value)


def test_llmtr_model_ignore_missing_cost(mock_response_no_cost):
    with patch.dict(os.environ, {"LLMTR_API_KEY": "test-llmtr-key"}):
        model = LLMTRModel(model_name="minimax/minimax-m3.1-flash", cost_tracking="ignore_errors")
        initial_cost = GLOBAL_MODEL_STATS.cost

        with patch("requests.post") as mock_post:
            mock_post.return_value.status_code = 200
            mock_post.return_value.json.return_value = mock_response_no_cost
            mock_post.return_value.raise_for_status.return_value = None

            result = model.query([{"role": "user", "content": "test"}])

            assert result["extra"]["cost"] == 0.0
            assert GLOBAL_MODEL_STATS.cost == initial_cost


def test_llmtr_model_config():
    with patch.dict(os.environ, {"LLMTR_API_KEY": "test-llmtr-key"}):
        model = LLMTRModel(model_name="minimax/minimax-m3.1-flash")
        assert model.config.model_name == "minimax/minimax-m3.1-flash"
        assert model._api_url == "https://llmtr.com/v1/chat/completions"
        assert model._api_key == "test-llmtr-key"
        assert "model_name" in model.get_template_vars()

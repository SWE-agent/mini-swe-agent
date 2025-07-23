import os
from unittest.mock import patch

from minisweagent.models.anthropic import AnthropicModel
from minisweagent.models.utils.key_per_thread import get_key_per_thread


def test_anthropic_model_single_key():
    with patch.dict(os.environ, {"ANTHROPIC_API_KEYS": "test-key"}):
        with patch("minisweagent.models.litellm_model.LitellmModel.query") as mock_query:
            mock_query.return_value = "response"

            model = AnthropicModel(model_name="tardis")
            result = model.query(messages=[])

            assert result == "response"
            assert mock_query.call_count == 1
            assert mock_query.call_args.kwargs["api_key"] == "test-key"


def test_get_key_per_thread_returns_same_key():
    key = get_key_per_thread(["1", "2"])
    for _ in range(100):
        assert get_key_per_thread(["1", "2"]) == key


def test_anthropic_model_with_empty_api_keys():
    with patch.dict(os.environ, {"ANTHROPIC_API_KEYS": ""}):
        with patch("minisweagent.models.litellm_model.LitellmModel.query") as mock_query:
            mock_query.return_value = "response"

            AnthropicModel(model_name="tardis").query(messages=[])

            assert mock_query.call_args.kwargs["api_key"] is None

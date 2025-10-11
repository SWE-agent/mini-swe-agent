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


def test_anthropic_model_applies_cache_control():
    """Test that AnthropicModel applies cache control to messages."""
    messages = [
        {"role": "user", "content": "Hello!"},
        {"role": "assistant", "content": "Hi there!"},
        {"role": "user", "content": "Help me code."},
    ]

    with patch("minisweagent.models.litellm_model.LitellmModel.query") as mock_query:
        mock_query.return_value = {"content": "I'll help you code!"}

        model = AnthropicModel(model_name="claude-sonnet")
        model.query(messages)

        # Verify parent query was called
        mock_query.assert_called_once()
        call_args = mock_query.call_args

        # Check that messages were modified with cache control
        passed_messages = call_args.args[0]  # messages is first positional arg

        # Only the last message should have cache control applied
        assert len(passed_messages) == 3

        # First two messages should not have cache control
        assert passed_messages[0]["content"] == "Hello!"
        assert passed_messages[1]["content"] == "Hi there!"

        # Last message should have cache control
        last_message = passed_messages[2]
        assert isinstance(last_message["content"], list)
        assert last_message["content"][0]["cache_control"] == {"type": "ephemeral"}
        assert last_message["content"][0]["type"] == "text"
        assert last_message["content"][0]["text"] == "Help me code."

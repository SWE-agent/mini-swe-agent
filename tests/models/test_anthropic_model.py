from unittest.mock import patch

from minisweagent.models.anthropic import AnthropicModel


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

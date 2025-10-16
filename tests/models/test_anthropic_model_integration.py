"""Test that cache control is actually applied when using anthropic models through get_model()."""

import copy
from unittest.mock import MagicMock, patch

import pytest

from minisweagent.models import get_model


def _mock_litellm_completion(response_content="Mock response"):
    """Helper to create consistent litellm mocks."""
    mock_response = MagicMock()
    mock_response.choices = [MagicMock()]
    mock_response.choices[0].message.content = response_content
    mock_response.model_dump.return_value = {"mock": "response"}
    return mock_response


def test_sonnet_4_cache_control_integration():
    """Test that get_model('sonnet-4') results in cache control being applied when querying."""
    messages = [
        {"role": "user", "content": "Hello, how are you?"},
        {"role": "assistant", "content": "I'm doing well!"},
        {"role": "user", "content": "Can you help me with coding?"},
    ]

    with patch("minisweagent.models.litellm_model.litellm.completion") as mock_completion:
        mock_completion.return_value = _mock_litellm_completion("Sure, I can help!")

        with patch("minisweagent.models.litellm_model.litellm.cost_calculator.completion_cost") as mock_cost:
            mock_cost.return_value = 0.001

            # This is the key test: get_model with anthropic name should enable cache control
            model = get_model("sonnet-4")
            model.query(messages)

            # Verify that cache control was applied to the messages sent to the API
            mock_completion.assert_called_once()
            call_kwargs = mock_completion.call_args.kwargs

            # Check the messages that were actually sent to litellm.completion
            sent_messages = call_kwargs["messages"]

            # Only the last message should have cache control applied
            assert len(sent_messages) == 3

            # First two messages should not have cache control
            assert sent_messages[0]["content"] == "Hello, how are you?"
            assert sent_messages[1]["content"] == "I'm doing well!"

            # Last message should have cache control
            last_message = sent_messages[2]
            assert isinstance(last_message["content"], list)
            assert last_message["content"][0]["cache_control"] == {"type": "ephemeral"}
            assert last_message["content"][0]["type"] == "text"
            assert last_message["content"][0]["text"] == "Can you help me with coding?"


@pytest.mark.parametrize(
    "model_name",
    [
        "sonnet-4",
        "claude-sonnet",
        "anthropic/claude",
        "opus-latest",
    ],
)
def test_get_model_anthropic_applies_cache_control(model_name):
    """Test that using get_model with anthropic model names results in cache control being applied."""
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Hello!"},
        {"role": "assistant", "content": "Hi there!"},
        {"role": "user", "content": "Help me code."},
    ]

    with patch("minisweagent.models.litellm_model.litellm.completion") as mock_completion:
        mock_completion.return_value = _mock_litellm_completion("I'll help you code!")

        with patch("minisweagent.models.litellm_model.litellm.cost_calculator.completion_cost") as mock_cost:
            mock_cost.return_value = 0.001

            # Get model through get_model - this should auto-configure cache control
            model = get_model(model_name)

            # Call query with a copy of messages (to avoid mutation issues)
            model.query(copy.deepcopy(messages))

            # Verify completion was called
            mock_completion.assert_called_once()
            call_args = mock_completion.call_args

            # Check that cache control was applied to the messages passed to litellm
            passed_messages = call_args.kwargs["messages"]

            # Only the last message should have cache control
            assert len(passed_messages) == 4, f"Expected 4 messages for {model_name}"

            # First three messages should not have cache control
            assert passed_messages[0]["content"] == "You are a helpful assistant.", (
                f"System message content should be preserved for {model_name}"
            )
            assert passed_messages[1]["content"] == "Hello!", (
                f"First user message content should be preserved for {model_name}"
            )
            assert passed_messages[2]["content"] == "Hi there!", (
                f"Assistant message content should be preserved for {model_name}"
            )

            # Last message should have cache control
            last_message = passed_messages[3]
            assert isinstance(last_message["content"], list), f"Last message should have list content for {model_name}"
            assert len(last_message["content"]) == 1, f"Last message should have single content item for {model_name}"

            content_item = last_message["content"][0]
            assert content_item["type"] == "text", f"Content should be text type for {model_name}"
            assert content_item["cache_control"] == {"type": "ephemeral"}, f"Cache control missing for {model_name}"
            assert content_item["text"] == "Help me code.", f"Text content should be preserved for {model_name}"


@pytest.mark.parametrize(
    "model_name",
    [
        "gpt-4",
        "gpt-3.5-turbo",
        "llama2",
    ],
)
def test_get_model_non_anthropic_no_cache_control(model_name):
    """Test that non-anthropic models don't get cache control applied."""
    messages = [
        {"role": "user", "content": "Hello!"},
    ]

    with patch("minisweagent.models.litellm_model.litellm.completion") as mock_completion:
        mock_completion.return_value = _mock_litellm_completion("Hello!")

        with patch("minisweagent.models.litellm_model.litellm.cost_calculator.completion_cost") as mock_cost:
            mock_cost.return_value = 0.001

            # Get model through get_model - should NOT auto-configure cache control
            model = get_model(model_name)

            # Call query
            model.query(copy.deepcopy(messages))

            # Verify completion was called
            mock_completion.assert_called_once()
            call_args = mock_completion.call_args

            # Check that messages were NOT modified with cache control
            passed_messages = call_args.kwargs["messages"]

            # The user message should still be a simple string, not transformed
            user_msg = passed_messages[0]
            assert user_msg["role"] == "user"
            assert user_msg["content"] == "Hello!", f"Content should remain as string for {model_name}"
            assert "cache_control" not in user_msg, f"No cache_control should be present for {model_name}"


def test_explicit_anthropic_model_class_cache_control():
    """Test that explicitly using AnthropicModel class also applies cache control."""
    from minisweagent.models.anthropic import AnthropicModel

    messages = [
        {"role": "user", "content": "Test message"},
    ]

    with patch("minisweagent.models.litellm_model.LitellmModel.query") as mock_query:
        mock_query.return_value = {"content": "Response"}

        # Create AnthropicModel directly
        model = AnthropicModel(model_name="claude-sonnet")
        model.query(copy.deepcopy(messages))

        # Verify parent query was called
        mock_query.assert_called_once()
        call_args = mock_query.call_args

        # Check that cache control was applied before calling parent
        passed_messages = call_args.args[0]  # messages is first positional arg

        user_msg = passed_messages[0]
        assert isinstance(user_msg["content"], list)
        assert user_msg["content"][0]["cache_control"] == {"type": "ephemeral"}
        assert user_msg["content"][0]["text"] == "Test message"

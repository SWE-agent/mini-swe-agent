from minisweagent.models.utils.cache_control import set_cache_control


def test_set_cache_control_basic():
    """Test basic cache control functionality with simple input/output."""
    # Input: A messages with multiple messages including user messages
    input_messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Hello, how are you?"},
        {"role": "assistant", "content": "I'm doing well, thank you!"},
        {"role": "user", "content": "Can you help me with coding?"},
        {"role": "assistant", "content": "Of course! I'd be happy to help."},
    ]

    # Expected output: Cache control added to the last 2 user messages
    expected_output = [
        {"role": "system", "content": "You are a helpful assistant."},
        {
            "role": "user",
            "content": [{"type": "text", "text": "Hello, how are you?", "cache_control": {"type": "ephemeral"}}],
        },
        {"role": "assistant", "content": "I'm doing well, thank you!"},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Can you help me with coding?", "cache_control": {"type": "ephemeral"}}
            ],
        },
        {"role": "assistant", "content": "Of course! I'd be happy to help."},
    ]

    result = set_cache_control(input_messages)

    assert result == expected_output


def test_set_cache_control_with_offset():
    """Test cache control with last_n_messages_offset parameter."""
    input_messages = [
        {"role": "user", "content": "First message"},
        {"role": "user", "content": "Second message"},
        {"role": "user", "content": "Third message"},
    ]

    # With offset=1, should skip the last message and tag the previous ones
    result = set_cache_control(input_messages, last_n_messages_offset=1)

    # Only the first two messages should have cache control
    assert "cache_control" not in result[2].get("content", {})  # Third message should not have cache control
    assert isinstance(result[0]["content"], list)  # First message should have cache control
    assert isinstance(result[1]["content"], list)  # Second message should have cache control

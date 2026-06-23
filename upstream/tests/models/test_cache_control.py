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

    # Expected output: Cache control added only to the last message
    expected_output = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Hello, how are you?"},
        {"role": "assistant", "content": "I'm doing well, thank you!"},
        {"role": "user", "content": "Can you help me with coding?"},
        {
            "role": "assistant",
            "content": [
                {"type": "text", "text": "Of course! I'd be happy to help.", "cache_control": {"type": "ephemeral"}}
            ],
        },
    ]

    result = set_cache_control(input_messages)

    assert result == expected_output


def test_set_cache_control_offset_deprecated():
    """Test that last_n_messages_offset parameter has no effect and is deprecated."""
    input_messages = [
        {"role": "user", "content": "First message"},
        {"role": "user", "content": "Second message"},
        {"role": "user", "content": "Third message"},
    ]

    # Test that offset parameter has no effect - should still only affect last message
    result_with_offset = set_cache_control(input_messages, last_n_messages_offset=1)
    result_without_offset = set_cache_control(input_messages)

    # Both results should be identical - offset should have no effect
    assert result_with_offset == result_without_offset

    # Only the last message should have cache control
    expected_output = [
        {"role": "user", "content": "First message"},
        {"role": "user", "content": "Second message"},
        {
            "role": "user",
            "content": [{"type": "text", "text": "Third message", "cache_control": {"type": "ephemeral"}}],
        },
    ]

    assert result_with_offset == expected_output
    assert result_without_offset == expected_output

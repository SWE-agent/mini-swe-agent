import copy

import pytest

from minisweagent.models.utils.cache_control import set_cache_control


@pytest.mark.parametrize(("role", "message_level"), [("user", False), ("tool", True)])
def test_set_cache_control_multiblock(role, message_level):
    content = [
        {"type": "text", "text": "Before"},
        {"type": "image_url", "image_url": {"url": "data:image/png;base64,test", "detail": "high"}},
        {"type": "text", "text": "After"},
    ]
    messages = [
        {
            "role": role,
            "content": [{**block, "cache_control": {"type": "ephemeral", "ttl": "1h"}} for block in content],
            "cache_control": {"type": "ephemeral"},
            "tool_call_id": "call_1",
            "extra": {"timestamp": 1},
        }
        for _ in range(2)
    ]
    original = copy.deepcopy(messages)
    expected = [{**message, "content": copy.deepcopy(content)} for message in messages]
    for message in expected:
        message.pop("cache_control")
    if message_level:
        expected[-1]["cache_control"] = {"type": "ephemeral"}
    else:
        expected[-1]["content"][-1]["cache_control"] = {"type": "ephemeral"}
    assert set_cache_control(messages) == expected
    assert set_cache_control(expected) == expected
    assert messages == original
    assert set_cache_control(messages, mode=None) is messages


@pytest.mark.parametrize(
    ("content", "cache_index"),
    [
        ([{"type": "image_url", "image_url": {"url": "https://example.com/image.png"}}], 0),
        ([{"type": "text", "text": "Answer"}, {"type": "thinking", "thinking": "Reasoning"}], 0),
        ([{"type": "text", "text": "Answer"}, {"type": "redacted_thinking", "data": "redacted"}], 0),
        ([{"type": "text", "text": "Answer"}, {"type": "text", "text": ""}], 0),
        ([{"type": "thinking", "thinking": "Reasoning"}, {"type": "text", "text": ""}], None),
        ([{"type": "redacted_thinking", "data": "redacted"}], None),
        ([], None),
    ],
)
def test_set_cache_control_last_cacheable_block(content, cache_index):
    messages = [{"role": "user", "content": content}]
    original = copy.deepcopy(messages)
    expected = copy.deepcopy(messages)
    if cache_index is not None:
        expected[0]["content"][cache_index]["cache_control"] = {"type": "ephemeral"}
    assert set_cache_control(messages) == expected
    assert messages == original


@pytest.mark.parametrize(
    ("content", "expected_content"),
    [
        (None, None),
        ("Output", [{"type": "text", "text": "Output"}]),
        ([{"type": "text", "text": "Output"}], [{"type": "text", "text": "Output"}]),
        ([], []),
    ],
)
def test_set_cache_control_tool_messages(content, expected_content):
    message = {"role": "tool", "content": content, "tool_call_id": "call_1"}
    assert set_cache_control([message]) == [
        {**message, "content": expected_content, "cache_control": {"type": "ephemeral"}}
    ]


def test_set_cache_control_none_content():
    message = {"role": "assistant", "content": None, "tool_calls": [{"id": "call_1", "type": "function"}]}
    assert set_cache_control([message]) == [{**message, "cache_control": {"type": "ephemeral"}}]


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

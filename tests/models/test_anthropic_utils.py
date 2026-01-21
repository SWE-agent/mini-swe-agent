import pytest

from minisweagent.models.utils.anthropic_utils import reorder_thinking_blocks


@pytest.mark.parametrize(
    ("messages", "expected"),
    [
        # No thinking blocks - unchanged
        (
            [{"role": "assistant", "content": [{"type": "text", "text": "hello"}]}],
            [{"role": "assistant", "content": [{"type": "text", "text": "hello"}]}],
        ),
        # Thinking block already first - unchanged
        (
            [
                {
                    "role": "assistant",
                    "content": [
                        {"type": "thinking", "thinking": "..."},
                        {"type": "text", "text": "hello"},
                    ],
                }
            ],
            [
                {
                    "role": "assistant",
                    "content": [
                        {"type": "thinking", "thinking": "..."},
                        {"type": "text", "text": "hello"},
                    ],
                }
            ],
        ),
        # Thinking block after text - reorder
        (
            [
                {
                    "role": "assistant",
                    "content": [
                        {"type": "text", "text": "hello"},
                        {"type": "thinking", "thinking": "..."},
                    ],
                }
            ],
            [
                {
                    "role": "assistant",
                    "content": [
                        {"type": "thinking", "thinking": "..."},
                        {"type": "text", "text": "hello"},
                    ],
                }
            ],
        ),
        # Multiple thinking blocks mixed
        (
            [
                {
                    "role": "assistant",
                    "content": [
                        {"type": "text", "text": "a"},
                        {"type": "thinking", "thinking": "1"},
                        {"type": "text", "text": "b"},
                        {"type": "redacted_thinking"},
                    ],
                }
            ],
            [
                {
                    "role": "assistant",
                    "content": [
                        {"type": "thinking", "thinking": "1"},
                        {"type": "redacted_thinking"},
                        {"type": "text", "text": "a"},
                        {"type": "text", "text": "b"},
                    ],
                }
            ],
        ),
        # Only thinking blocks - adds empty text block
        (
            [{"role": "assistant", "content": [{"type": "thinking", "thinking": "..."}]}],
            [
                {
                    "role": "assistant",
                    "content": [{"type": "thinking", "thinking": "..."}, {"type": "text", "text": ""}],
                }
            ],
        ),
        # Non-assistant message - unchanged
        (
            [{"role": "user", "content": [{"type": "text", "text": "hello"}]}],
            [{"role": "user", "content": [{"type": "text", "text": "hello"}]}],
        ),
        # String content - unchanged
        (
            [{"role": "assistant", "content": "hello"}],
            [{"role": "assistant", "content": "hello"}],
        ),
        # Preserves other message fields
        (
            [
                {
                    "role": "assistant",
                    "content": [{"type": "text", "text": "hi"}, {"type": "thinking", "thinking": "..."}],
                    "extra_field": "preserved",
                }
            ],
            [
                {
                    "role": "assistant",
                    "content": [{"type": "thinking", "thinking": "..."}, {"type": "text", "text": "hi"}],
                    "extra_field": "preserved",
                }
            ],
        ),
    ],
)
def test_reorder_thinking_blocks(messages, expected):
    assert reorder_thinking_blocks(messages) == expected

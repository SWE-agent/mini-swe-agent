import pytest

from minisweagent.models.utils.gemini_utils import _deduplicate_gemini_thought_signatures


@pytest.mark.parametrize(
    ("messages", "expected"),
    [
        # Assistant message with both top-level and tool_call provider_specific_fields - strips top-level
        (
            [
                {
                    "role": "assistant",
                    "content": "Running command",
                    "provider_specific_fields": {"thought_signatures": ["sig1"]},
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "type": "function",
                            "function": {"name": "bash", "arguments": "ls"},
                            "provider_specific_fields": {"thought_signature": "sig1"},
                        }
                    ],
                }
            ],
            [
                {
                    "role": "assistant",
                    "content": "Running command",
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "type": "function",
                            "function": {"name": "bash", "arguments": "ls"},
                            "provider_specific_fields": {"thought_signature": "sig1"},
                        }
                    ],
                }
            ],
        ),
        # Assistant message with top-level provider_specific_fields but no tool calls - preserved
        (
            [
                {
                    "role": "assistant",
                    "content": "Just text",
                    "provider_specific_fields": {"thought_signatures": ["sig1"]},
                }
            ],
            [
                {
                    "role": "assistant",
                    "content": "Just text",
                    "provider_specific_fields": {"thought_signatures": ["sig1"]},
                }
            ],
        ),
        # Assistant message with tool calls without provider_specific_fields - unchanged
        (
            [
                {
                    "role": "assistant",
                    "content": "Running command",
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "type": "function",
                            "function": {"name": "bash", "arguments": "ls"},
                        }
                    ],
                }
            ],
            [
                {
                    "role": "assistant",
                    "content": "Running command",
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "type": "function",
                            "function": {"name": "bash", "arguments": "ls"},
                        }
                    ],
                }
            ],
        ),
        # User or system messages - unchanged
        (
            [
                {"role": "system", "content": "You are helpful"},
                {"role": "user", "content": "Hello", "provider_specific_fields": {"test": 123}},
            ],
            [
                {"role": "system", "content": "You are helpful"},
                {"role": "user", "content": "Hello", "provider_specific_fields": {"test": 123}},
            ],
        ),
    ],
)
def test_deduplicate_gemini_thought_signatures(messages, expected):
    assert _deduplicate_gemini_thought_signatures(messages) == expected


def test_litellm_gemini_payload_deduplication():
    """Demonstrate before/after effect of deduplication on LiteLLM's Gemini API transformation."""
    try:
        from litellm.llms.vertex_ai.gemini.vertex_and_google_ai_studio_gemini import (
            VertexGeminiConfig,
        )
    except ImportError:
        pytest.skip("LiteLLM Gemini module not available")

    config = VertexGeminiConfig()

    raw_message = {
        "role": "assistant",
        "content": "I will list the directory.",
        "provider_specific_fields": {"thought_signatures": ["mock_signature_63k"]},
        "tool_calls": [
            {
                "id": "call_1",
                "type": "function",
                "function": {"name": "bash", "arguments": '{"command": "ls -la"}'},
                "provider_specific_fields": {"thought_signature": "mock_signature_63k"},
            }
        ],
    }

    # Before fix: LiteLLM transforms raw message into duplicate thoughtSignature blocks across parts
    before_parts = config._transform_messages(messages=[raw_message], model="gemini/earhart")[0]["parts"]
    assert len(before_parts) == 2
    assert before_parts[0] == {"text": "I will list the directory.", "thoughtSignature": "mock_signature_63k"}
    assert "thoughtSignature" in before_parts[1]  # Duplicate thought signature attached to function_call!

    # After fix: deduplication cleans the message before LiteLLM transformation
    after_messages = _deduplicate_gemini_thought_signatures([raw_message])
    after_parts = config._transform_messages(messages=after_messages, model="gemini/earhart")[0]["parts"]
    assert len(after_parts) == 2
    assert after_parts[0] == {"text": "I will list the directory."}  # Clean text part without duplicate signature!
    assert "thoughtSignature" in after_parts[1]  # Thought signature correctly retained solely on function_call

"""Utilities for Anthropic API compatibility."""


def _is_anthropic_thinking_block(block) -> bool:
    """Check if a content block is a thinking-type block."""
    if not isinstance(block, dict):
        return False
    return block.get("type") in ("thinking", "redacted_thinking")


def _reorder_anthropic_thinking_blocks(messages: list[dict]) -> list[dict]:
    """Reorder thinking blocks so they are not the final block in assistant messages.

    This is an Anthropic API requirement: thinking blocks must come before other blocks.
    """
    result = []
    for msg in messages:
        if msg.get("role") == "assistant" and isinstance(msg.get("content"), list):
            content = msg["content"]
            thinking_blocks = [b for b in content if _is_anthropic_thinking_block(b)]
            if thinking_blocks:
                other_blocks = [b for b in content if not _is_anthropic_thinking_block(b)]
                if other_blocks:
                    msg = {**msg, "content": thinking_blocks + other_blocks}
                else:
                    msg = {**msg, "content": thinking_blocks + [{"type": "text", "text": ""}]}
        result.append(msg)
    return result

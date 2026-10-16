"""Cache control utilities are mostly for Anthropic models.
They are used to explicitly set cache control points.
"""

import copy
import warnings
from typing import Literal

from minisweagent.models.utils.anthropic_utils import _is_anthropic_thinking_block


def _clear_cache_control(entry: dict) -> None:
    if isinstance(entry["content"], list):
        for block in entry["content"]:
            block.pop("cache_control", None)
    # Note: entry["content"] can be None for assistant messages with only tool_use
    entry.pop("cache_control", None)


def _set_cache_control(entry: dict) -> None:
    # Handle None content (e.g., assistant messages with only tool_use)
    if entry["content"] is None:
        entry["cache_control"] = {"type": "ephemeral"}
        return

    if not isinstance(entry["content"], list):
        entry["content"] = [{"type": "text", "text": entry["content"]}]
    if entry["role"] == "tool":
        # Workaround for weird bug
        entry["cache_control"] = {"type": "ephemeral"}
        return
    for block in reversed(entry["content"]):
        if _is_anthropic_thinking_block(block) or (block.get("type") == "text" and not block.get("text")):
            continue
        block["cache_control"] = {"type": "ephemeral"}
        break


def set_cache_control(
    messages: list[dict], *, mode: Literal["default_end"] | None = "default_end", last_n_messages_offset: int = 0
) -> list[dict]:
    """This messages processor adds manual cache control marks to the messages."""
    if mode is None:
        return messages
    if mode != "default_end":
        raise ValueError(f"Invalid mode: {mode}")
    if last_n_messages_offset:
        warnings.warn("last_n_messages_offset is deprecated and will be removed in the future. It has no effect.")

    messages = copy.deepcopy(messages)
    new_messages = []
    for i_entry, entry in enumerate(reversed(messages)):
        _clear_cache_control(entry)
        if i_entry == 0:
            _set_cache_control(entry)
        new_messages.append(entry)
    return list(reversed(new_messages))

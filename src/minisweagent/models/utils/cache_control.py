import copy
import warnings
from typing import Literal


def _get_content_text(entry: dict) -> str:
    if isinstance(entry["content"], str):
        return entry["content"]
    assert len(entry["content"]) == 1, "Expected single message in content"
    return entry["content"][0]["text"]


def _clear_cache_control(entry: dict) -> None:
    if isinstance(entry["content"], list):
        assert len(entry["content"]) == 1, "Expected single message in content"
        entry["content"][0].pop("cache_control", None)
    entry.pop("cache_control", None)


def _set_cache_control(entry: dict) -> None:
    if not isinstance(entry["content"], list):
        entry["content"] = [  # type: ignore
            {
                "type": "text",
                "text": _get_content_text(entry),
                "cache_control": {"type": "ephemeral"},
            }
        ]
    else:
        entry["content"][0]["cache_control"] = {"type": "ephemeral"}
    if entry["role"] == "tool":
        # Workaround for weird bug
        entry["content"][0].pop("cache_control", None)
        entry["cache_control"] = {"type": "ephemeral"}


def set_cache_control(
    messages: list[dict], *, mode: Literal["default_end"] | None = "default_end", last_n_messages_offset: int = 0
) -> list[dict]:
    """This messages processor adds manual cache control marks to the messages."""
    if mode != "default_end":
        raise ValueError(f"Invalid mode: {mode}")
    if last_n_messages_offset:
        warnings.warn("last_n_messages_offset is deprecated and will be removed in the future")
    messages = copy.deepcopy(messages)

    # Find all user messages
    user_message_indices = []
    for i, msg in enumerate(messages):
        if msg["role"] == "user":
            user_message_indices.append(i)

    # Clear cache control from all messages first
    for entry in messages:
        _clear_cache_control(entry)

    # Add cache control to user messages, respecting offset
    if user_message_indices:
        # If offset is specified, exclude the last N messages from getting cache control
        messages_to_cache = user_message_indices
        if last_n_messages_offset > 0:
            messages_to_cache = (
                user_message_indices[:-last_n_messages_offset]
                if last_n_messages_offset < len(user_message_indices)
                else []
            )

        # Add cache control to the selected user messages
        for idx in messages_to_cache:
            _set_cache_control(messages[idx])

    return messages

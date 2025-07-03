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


def set_cache_control(messages: list[dict], last_n_messages_offset: int = 0) -> list[dict]:
    """This messages processor adds manual cache control marks to the messages."""
    new_messages = []
    n_tagged = 0
    for i_entry, entry in enumerate(reversed(messages)):
        _clear_cache_control(entry)
        if n_tagged < 2 and entry["role"] in ["user"] and i_entry >= last_n_messages_offset:
            _set_cache_control(entry)
            n_tagged += 1
        new_messages.append(entry)
    return list(reversed(new_messages))

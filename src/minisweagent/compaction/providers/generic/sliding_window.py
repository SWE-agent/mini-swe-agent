"""Sliding window compaction: keeps system + first user message + the last N steps."""


def _group_into_steps(messages: list[dict]) -> list[list[dict]]:
    """Group messages[2:] into steps: each step is one assistant msg + its tool results."""
    steps: list[list[dict]] = []
    current: list[dict] = []
    for msg in messages:
        if msg.get("role") == "assistant" and current:
            steps.append(current)
            current = [msg]
        else:
            current.append(msg)
    if current:
        steps.append(current)
    return steps


class SlidingWindowCompaction:
    """Keep only the last `keep_last_n_steps` conversation steps.

    Always preserves messages[0] (system) and messages[1] (first user/task).
    """

    def __init__(self, keep_last_n_steps: int = 10):
        self.keep_last_n_steps = keep_last_n_steps

    def compact(self, messages: list[dict]) -> list[dict]:
        if len(messages) <= 2:
            return messages
        anchors = messages[:2]
        steps = _group_into_steps(messages[2:])
        kept_steps = steps[-self.keep_last_n_steps :]
        kept_messages = [msg for step in kept_steps for msg in step]
        return anchors + kept_messages

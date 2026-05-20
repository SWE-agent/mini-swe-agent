class NoopCompaction:
    """Pass-through strategy — no compaction applied."""

    def compact(self, messages: list[dict]) -> list[dict]:
        return messages

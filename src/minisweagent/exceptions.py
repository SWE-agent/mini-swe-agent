class InterruptAgentFlow(Exception):
    """Raised to interrupt the agent flow and add messages."""

    def __init__(self, *messages: dict):
        self.messages = messages
        super().__init__()


class Submitted(InterruptAgentFlow):
    """Raised when the agent has completed its task."""


class LimitsExceeded(InterruptAgentFlow):
    """Raised when the agent has exceeded its cost or step limit."""


class TimeExceeded(LimitsExceeded):
    """Raised when the agent has exceeded its wall-clock time limit."""


class UserInterruption(InterruptAgentFlow):
    """Raised when the user interrupts the agent."""


class FormatError(InterruptAgentFlow):
    """Raised when the LM's output is not in the expected format."""


class ContextWindowExceeded(Exception):
    """Raised when a model request exceeds the provider's context window."""


def is_context_window_error(error: object) -> bool:
    """Return whether an arbitrary provider error describes context overflow."""
    if isinstance(error, dict):
        values = [error, error.get("error"), error.get("metadata")]
        text = " ".join(str(value) for value in values if value).lower()
    else:
        code = getattr(error, "code", "")
        body = getattr(error, "body", "")
        text = f"{code} {body} {error}".lower()
    return "context_length_exceeded" in text or (
        ("context" in text or "token" in text)
        and ("length" in text or "window" in text or "limit" in text or "maximum" in text)
        and ("exceed" in text or "too long" in text or "maximum" in text or "limit" in text)
    )

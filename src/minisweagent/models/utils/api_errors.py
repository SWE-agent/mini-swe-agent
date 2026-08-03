"""Helpers for classifying provider API errors."""


def is_context_length_error(status_code: int, response_text: str) -> bool:
    """Return whether a 400 response identifies a permanent context overflow."""
    if status_code != 400:
        return False
    response_text = response_text.lower()
    return "context_length_exceeded" in response_text or "context_window_exceeded" in response_text

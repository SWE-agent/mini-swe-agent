"""Utilities for classifying model API errors."""

import re

import requests

_CONTEXT_WINDOW_ERROR_CODE = "context_length_exceeded"
_CONTEXT_WINDOW_ERROR_MESSAGES = (
    "context length exceeded",
    "context window exceeded",
    "context limit exceeded",
    "exceeds the context window",
    "maximum context length",
    "context window exceeds limit",
)


def is_context_window_error(response: requests.Response) -> bool:
    """Return whether a 400 response reports a context window overflow."""
    if response.status_code != 400:
        return False
    try:
        body = response.json()
    except requests.exceptions.JSONDecodeError:
        body = {}
    body = body if isinstance(body, dict) else {}
    error = body.get("error", {})
    error = error if isinstance(error, dict) else {}
    metadata = error.get("metadata", {})
    metadata = metadata if isinstance(metadata, dict) else {}
    codes = (
        body.get("error_type"),
        error.get("code"),
        error.get("type"),
        error.get("error_type"),
        metadata.get("error_type"),
        metadata.get("provider_code"),
    )
    if _CONTEXT_WINDOW_ERROR_CODE in codes:
        return True
    message = error.get("message")
    if not isinstance(message, str):
        return False
    message = " ".join(re.findall(r"[a-z0-9]+", message.lower()))
    return any(fragment in message for fragment in _CONTEXT_WINDOW_ERROR_MESSAGES)

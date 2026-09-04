"""Utilities for preparing execution observations for model providers."""

MAX_OBSERVATION_OUTPUT_BYTES = 2000


def truncate_observation_text(text: object, *, max_bytes: int = MAX_OBSERVATION_OUTPUT_BYTES) -> str:
    """Bound provider-facing command output while preserving both ends."""
    text = "" if text is None else str(text)
    encoded = text.encode("utf-8")
    if len(encoded) <= max_bytes:
        return text

    omitted = len(encoded) - max_bytes
    while True:
        marker = f"\n...[output truncated, {omitted} bytes omitted]...\n"
        marker_bytes = len(marker.encode("utf-8"))
        remaining = max(max_bytes - marker_bytes, 0)
        new_omitted = len(encoded) - remaining
        if new_omitted == omitted:
            break
        omitted = new_omitted

    if marker_bytes > max_bytes:
        return marker.encode("utf-8")[:max_bytes].decode("utf-8", errors="ignore")

    remaining = max(max_bytes - marker_bytes, 0)
    head_bytes = remaining // 2
    tail_bytes = remaining - head_bytes

    head = encoded[:head_bytes].decode("utf-8", errors="ignore")
    tail = encoded[-tail_bytes:].decode("utf-8", errors="ignore") if tail_bytes else ""
    return f"{head}{marker}{tail}"


def bounded_observation_output(output: dict) -> dict:
    """Return a shallow copy with only the provider-facing output truncated."""
    bounded = dict(output)
    bounded["output"] = truncate_observation_text(bounded.get("output", ""))
    return bounded

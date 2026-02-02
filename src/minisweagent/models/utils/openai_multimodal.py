"""Utilities for handling multimodal content in OpenAI-style messages."""

import copy
import re
from typing import Any

DEFAULT_MULTIMODAL_REGEX = (
    r"(?s)<MSWEA_MULTIMODAL_CONTENT><CONTENT_TYPE>(.+?)</CONTENT_TYPE>(.+?)</MSWEA_MULTIMODAL_CONTENT>"
)


def _expand_content_string(*, content: str, pattern: str) -> list[dict]:
    """Expand a content string, replacing multimodal tags with structured content."""
    matches = list(re.finditer(pattern, content))
    if not matches:
        return [{"type": "text", "text": content}]
    result = []
    last_end = 0
    for match in matches:
        text_before = content[last_end : match.start()]
        if text_before:
            result.append({"type": "text", "text": text_before})
        content_type = match.group(1).strip()
        extracted = match.group(2).strip()
        if content_type == "image_url":
            result.append({"type": "image_url", "image_url": {"url": extracted}})
        last_end = match.end()
    text_after = content[last_end:]
    if text_after:
        result.append({"type": "text", "text": text_after})
    return result


def expand_multimodal_content(content: Any, *, pattern: str) -> Any:
    """Recursively expand multimodal content in messages.
    Note: Returns copy of content, original content is not modified.
    """
    if not pattern:
        return content
    content = copy.deepcopy(content)
    if isinstance(content, str):
        return _expand_content_string(content=content, pattern=pattern)
    if isinstance(content, list):
        return [expand_multimodal_content(item, pattern=pattern) for item in content]
    if isinstance(content, dict):
        if "content" not in content:
            return content
        content["content"] = expand_multimodal_content(content["content"], pattern=pattern)
        return content
    return str(content)

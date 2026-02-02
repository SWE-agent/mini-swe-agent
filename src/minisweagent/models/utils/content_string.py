"""Helper function for pretty-printing content strings."""

import json


def _format_tool_call(args_str: str) -> str:
    """Format tool call arguments, extracting command if it's a bash call."""
    try:
        args = json.loads(args_str) if isinstance(args_str, str) else args_str
        if isinstance(args, dict) and "command" in args:
            return f"```\n{args['command']}\n```"
    except Exception:
        pass
    return f"```\n{args_str}\n```"


def _format_observation(content: str) -> str | None:
    """Try to format an observation JSON as key-value pairs."""
    try:
        data = json.loads(content)
        if isinstance(data, dict) and "returncode" in data:
            lines = []
            for key, value in data.items():
                lines.append(f"<{key}>")
                lines.append(str(value))
            return "\n".join(lines)
        return content
    except Exception:
        return content


def get_content_string(message: dict) -> str:
    """Extract text content from any message format for display.

    Handles:
    - Traditional chat: {"content": "text"}
    - Multimodal chat: {"content": [{"type": "text", "text": "..."}]}
    - Observation messages: {"content": "{\"returncode\": 0, \"output\": \"...\"}"}
    - Traditional tool calls: {"tool_calls": [{"function": {"name": "...", "arguments": "..."}}]}
    - Responses API: {"output": [{"type": "message", "content": [...]}]}
    """
    texts = []

    # Extract content (string or multimodal list)
    content = message.get("content")
    if isinstance(content, str):
        texts.append(_format_observation(content))
    elif isinstance(content, list):
        texts.append("\n".join(item.get("text", "") for item in content if isinstance(item, dict)))

    # Handle traditional tool_calls format (OpenAI/LiteLLM style)
    if tool_calls := message.get("tool_calls"):
        for tc in tool_calls:
            func = tc.get("function", {}) if isinstance(tc, dict) else getattr(tc, "function", None)
            if func:
                args = func.get("arguments", "{}") if isinstance(func, dict) else getattr(func, "arguments", "{}")
                texts.append(_format_tool_call(args))

    # Handle Responses API format (output array)
    if output := message.get("output"):
        if isinstance(output, str):
            texts.append(_format_observation(output))
        elif isinstance(output, list):
            for item in output:
                if not isinstance(item, dict):
                    continue
                if item.get("type") == "message":
                    for c in item.get("content", []):
                        if isinstance(c, dict) and (text := c.get("text")):
                            texts.append(text)
                elif item.get("type") == "function_call":
                    texts.append(_format_tool_call(item.get("arguments", "{}")))

    return "\n\n".join(t for t in texts if t)

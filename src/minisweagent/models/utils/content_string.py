def coerce_message_content(message: dict) -> str:
    """Extract text content from any message format for display.

    Handles:
    - Traditional chat: {"content": "text"}
    - Multimodal chat: {"content": [{"type": "text", "text": "..."}]}
    - Responses API: {"output": [{"type": "message", "content": [...]}]}
    """
    # Try traditional content field first
    content = message.get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(item.get("text", "") for item in content if isinstance(item, dict))

    # Try Responses API format (output array)
    if output := message.get("output"):
        texts = []
        for item in output:
            if not isinstance(item, dict):
                continue
            # Handle message items with nested content
            if item.get("type") == "message":
                for c in item.get("content", []):
                    if isinstance(c, dict) and (text := c.get("text")):
                        texts.append(text)
            # Handle function_call items (show as formatted call)
            elif item.get("type") == "function_call":
                name = item.get("name", "unknown")
                args = item.get("arguments", "{}")
                texts.append(f"Tool call: {name}({args})")
        if texts:
            return "\n\n".join(texts)

    return str(content) if content else ""

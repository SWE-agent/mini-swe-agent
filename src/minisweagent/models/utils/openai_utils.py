import logging
from typing import Any

from openai.types.responses.response_output_message import ResponseOutputMessage

logger = logging.getLogger("openai_utils")


def coerce_responses_text(resp: Any) -> str:
    """Helper to normalize OpenAI Responses API result to text.

    Works with both OpenAI client responses and LiteLLM/Portkey responses.
    """
    text = getattr(resp, "output_text", None)
    if isinstance(text, str) and text:
        return text

    try:
        output = []
        for item in resp.output:
            if isinstance(item, dict):
                content = item.get("content", [])
            elif isinstance(item, ResponseOutputMessage):
                content = item.content
            else:
                continue

            for content_item in content:
                if isinstance(content_item, dict):
                    text_val = content_item.get("text")
                elif hasattr(content_item, "text"):
                    text_val = content_item.text
                else:
                    continue

                if text_val:
                    output.append(text_val)
        return "\n\n".join(output) or ""
    except (AttributeError, IndexError, TypeError):
        logger.warning(f"Could not extract text from response: {resp}")
        return ""

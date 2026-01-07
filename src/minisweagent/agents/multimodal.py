"""This class extends the DefaultAgent class to support images.
The idea here is super simple: Every time we encounter image data (URL or encoded data)
within <MSWEA_IMG_CONTENT>...</MSWEA_IMG_CONTENT> tags, we expand it to an image_url message.
So all we need to do is to override DefaultAgent.add_messages.
"""

import copy
import re
from typing import Any

from minisweagent.agents.default import AgentConfig, DefaultAgent


class MultimodalAgentConfig(AgentConfig):
    image_regex: str = r"(?s)<MSWEA_IMG_CONTENT>(.{10,}?)</MSWEA_IMG_CONTENT>"
    """Regex to extract the image from the content. Requires at least 10 characters
    so that we can still reference it in the prompts. Matches multiline content.
    """


def _expand_content_string(*, content: str, pattern: str) -> list[dict]:
    matches = list(re.finditer(pattern, content, re.DOTALL))
    if not matches:
        return [{"type": "text", "content": content}]
    result = []
    last_end = 0
    for match in matches:
        text_before = content[last_end : match.start()]
        if text_before:
            result.append({"type": "text", "content": text_before})
        result.append({"type": "image_url", "image_url": {"url": match.group(1).strip()}})
        last_end = match.end()
    text_after = content[last_end:]
    if text_after:
        result.append({"type": "text", "content": text_after})
    return result


class MultimodalAgent(DefaultAgent):
    def __init__(self, *args, config_class: type = MultimodalAgentConfig, **kwargs):
        super().__init__(*args, config_class=config_class, **kwargs)

    def _expand_content(self, content: Any) -> Any:
        content = copy.deepcopy(content)
        if isinstance(content, str):
            return _expand_content_string(content=content, pattern=self.config.image_regex)
        if isinstance(content, list):
            return [self._expand_content(item) for item in content]
        if isinstance(content, dict):
            if "content" not in content:
                return content
            content["content"] = self._expand_content(content["content"])
            return content
        return str(content)

    def add_messages(self, *messages: dict) -> list[dict]:
        messages = [self._expand_content(msg) for msg in messages]
        return super().add_messages(*messages)

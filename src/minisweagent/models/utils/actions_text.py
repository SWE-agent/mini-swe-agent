"""Parse actions & format observations without toolcalls"""

import re
import time

from jinja2 import StrictUndefined, Template

from minisweagent.exceptions import FormatError
from minisweagent.models.utils.openai_multimodal import expand_multimodal_content


def parse_regex_actions(content: str, *, action_regex: str, format_error_template: str) -> list[dict]:
    """Parse actions from text content using regex. Raises FormatError if not exactly one action."""
    actions = [a.strip() for a in re.findall(action_regex, content, re.DOTALL)]
    if len(actions) != 1:
        raise FormatError(
            {
                "role": "user",
                "content": Template(format_error_template, undefined=StrictUndefined).render(actions=actions),
                "extra": {
                    "interrupt_type": "FormatError",
                    "n_actions": len(actions),
                    "model_response": content,
                },
            }
        )
    return [{"command": action} for action in actions]


def _build_observation_extra(output: dict) -> dict:
    """Build the 'extra' dict for an observation message."""
    extra = {
        "raw_output": output.get("output", ""),
        "returncode": output.get("returncode"),
        "timestamp": time.time(),
    }
    if output.get("exception_info"):
        extra["exception_info"] = output["exception_info"]
        extra.update(output.get("extra", {}))
    return extra


def format_observation_messages(
    outputs: list[dict],
    *,
    observation_template: str,
    template_vars: dict | None = None,
    multimodal_regex: str = "",
) -> list[dict]:
    """Format execution outputs into user observation messages."""
    results = []
    for output in outputs:
        content = Template(observation_template, undefined=StrictUndefined).render(
            output=output, **(template_vars or {})
        )
        msg: dict = {"role": "user", "content": content, "extra": _build_observation_extra(output)}
        if multimodal_regex:
            msg = expand_multimodal_content(msg, multimodal_regex)
        results.append(msg)
    return results

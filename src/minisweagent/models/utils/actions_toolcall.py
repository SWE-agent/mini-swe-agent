"""Parse actions & format observations with toolcalls"""

import json
import time
from collections.abc import Sequence
from typing import Any

from jinja2 import StrictUndefined, Template

from minisweagent.exceptions import FormatError
from minisweagent.models.utils.openai_multimodal import expand_multimodal_content

BASH_TOOL = {
    "type": "function",
    "function": {
        "name": "bash",
        "description": "Execute a bash command",
        "parameters": {
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "description": "The bash command to execute",
                }
            },
            "required": ["command"],
        },
    },
}


def _message_to_dict(message: Any) -> dict:
    if hasattr(message, "model_dump"):
        return message.model_dump()
    if isinstance(message, dict):
        return dict(message)
    return {
        "role": getattr(message, "role", "assistant"),
        "content": getattr(message, "content", None),
        "tool_calls": getattr(message, "tool_calls", None),
    }


def _tool_call_attr(tool_call: Any, key: str) -> Any:
    if isinstance(tool_call, dict):
        return tool_call.get(key)
    return getattr(tool_call, key, None)


def _tool_function_attr(tool_call: Any, key: str) -> Any:
    function = _tool_call_attr(tool_call, "function")
    if isinstance(function, dict):
        return function.get(key)
    return getattr(function, key, None)


def merge_choice_messages(choices: Sequence[Any]) -> dict:
    """Merge assistant content/tool calls across multiple choices into one message dict."""
    merged: dict[str, Any] = {}
    contents: list[Any] = []
    tool_calls: list[Any] = []

    for choice in choices:
        choice_message = _message_to_dict(choice.message if hasattr(choice, "message") else choice["message"])
        if not merged:
            merged = dict(choice_message)
        content = choice_message.get("content")
        if content not in (None, "", []):
            contents.append(content)
        if choice_message.get("tool_calls"):
            tool_calls.extend(choice_message["tool_calls"])

    if not merged:
        return {"role": "assistant", "content": None, "tool_calls": []}

    if not contents:
        merged["content"] = None
    elif len(contents) == 1:
        merged["content"] = contents[0]
    elif all(isinstance(content, str) for content in contents):
        merged["content"] = "\n".join(content for content in contents if content)
    else:
        merged_content = []
        for content in contents:
            if isinstance(content, list):
                merged_content.extend(content)
            else:
                merged_content.append(content)
        merged["content"] = merged_content

    merged["tool_calls"] = tool_calls
    return merged


def parse_toolcall_actions(tool_calls: list, *, format_error_template: str) -> list[dict]:
    """Parse tool calls from the response. Raises FormatError if unknown tool or invalid args."""
    if not tool_calls:
        raise FormatError(
            {
                "role": "user",
                "content": Template(format_error_template, undefined=StrictUndefined).render(
                    error="No tool calls found in the response. Every response MUST include at least one tool call.",
                    actions=[],
                ),
                "extra": {"interrupt_type": "FormatError"},
            }
        )
    actions = []
    for tool_call in tool_calls:
        error_msg = ""
        args = {}
        tool_name = _tool_function_attr(tool_call, "name")
        tool_arguments = _tool_function_attr(tool_call, "arguments")
        tool_call_id = _tool_call_attr(tool_call, "id")
        try:
            args = json.loads(tool_arguments)
        except Exception as e:
            error_msg = f"Error parsing tool call arguments: {e}."
        if tool_name != "bash":
            error_msg += f"Unknown tool '{tool_name}'."
        if not isinstance(args, dict) or "command" not in args:
            error_msg += "Missing 'command' argument in bash tool call."
        if error_msg:
            raise FormatError(
                {
                    "role": "user",
                    "content": Template(format_error_template, undefined=StrictUndefined).render(
                        actions=[], error=error_msg.strip()
                    ),
                    "extra": {"interrupt_type": "FormatError"},
                }
            )
        actions.append({"command": args["command"], "tool_call_id": tool_call_id})
    return actions


def format_toolcall_observation_messages(
    *,
    actions: list[dict],
    outputs: list[dict],
    observation_template: str,
    template_vars: dict | None = None,
    multimodal_regex: str = "",
) -> list[dict]:
    """Format execution outputs into tool result messages."""
    not_executed = {"output": "", "returncode": -1, "exception_info": "action was not executed"}
    padded_outputs = outputs + [not_executed] * (len(actions) - len(outputs))
    results = []
    for action, output in zip(actions, padded_outputs):
        content = Template(observation_template, undefined=StrictUndefined).render(
            output=output, **(template_vars or {})
        )
        msg = {
            "content": content,
            "extra": {
                "raw_output": output.get("output", ""),
                "returncode": output.get("returncode"),
                "timestamp": time.time(),
                "exception_info": output.get("exception_info"),
                **output.get("extra", {}),
            },
        }
        if "tool_call_id" in action:
            msg["tool_call_id"] = action["tool_call_id"]
            msg["role"] = "tool"
        else:
            msg["role"] = "user"  # human issued commands
        if multimodal_regex:
            msg = expand_multimodal_content(msg, pattern=multimodal_regex)
        results.append(msg)
    return results

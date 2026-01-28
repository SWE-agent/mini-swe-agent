"""Parse actions & format observations for OpenAI Responses API toolcalls"""

import json
import time

from jinja2 import StrictUndefined, Template

from minisweagent.exceptions import FormatError

# OpenRouter/OpenAI Responses API uses a flat structure (no nested "function" key)
BASH_TOOL_RESPONSE_API = {
    "type": "function",
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
}


def _format_error_message(error_text: str) -> dict:
    """Create a FormatError message in Responses API format."""
    return {
        "type": "message",
        "role": "user",
        "content": [{"type": "input_text", "text": error_text}],
        "extra": {"interrupt_type": "FormatError"},
    }


def parse_toolcall_actions_response(tool_calls: list[dict], *, format_error_template: str) -> list[dict]:
    """Parse tool calls from a Responses API response.

    Response API format has name/arguments at top level with call_id:
    {"type": "function_call", "call_id": "...", "name": "bash", "arguments": "..."}
    """
    if not tool_calls:
        error_text = Template(format_error_template, undefined=StrictUndefined).render(
            error="No tool calls found in the response. Every response MUST include at least one tool call.",
        )
        raise FormatError(_format_error_message(error_text))
    actions = []
    for tool_call in tool_calls:
        error_msg = ""
        args = {}
        try:
            args = json.loads(tool_call.get("arguments", "{}"))
        except Exception as e:
            error_msg = f"Error parsing tool call arguments: {e}. "
        if tool_call.get("name") != "bash":
            error_msg += f"Unknown tool '{tool_call.get('name')}'."
        if "command" not in args:
            error_msg += "Missing 'command' argument in bash tool call."
        if error_msg:
            error_text = Template(format_error_template, undefined=StrictUndefined).render(error=error_msg.strip())
            raise FormatError(_format_error_message(error_text))
        actions.append({"command": args["command"], "tool_call_id": tool_call.get("call_id") or tool_call.get("id")})
    return actions


def format_toolcall_observation_messages(
    *,
    actions: list[dict],
    outputs: list[dict],
    observation_template: str,
    template_vars: dict | None = None,
    multimodal_regex: str = "",
) -> list[dict]:
    """Format execution outputs into function_call_output messages for Responses API."""
    results = []
    for action, output in zip(actions, outputs):
        content = Template(observation_template, undefined=StrictUndefined).render(
            output=output, **(template_vars or {})
        )
        extra = {
            "raw_output": output.get("output", ""),
            "returncode": output.get("returncode"),
            "timestamp": time.time(),
        }
        if output.get("exception_info"):
            extra["exception_info"] = output["exception_info"]
            extra.update(output.get("extra", {}))
        results.append(
            {
                "type": "function_call_output",
                "call_id": action["tool_call_id"],
                "output": content,
                "extra": extra,
            }
        )
    return results

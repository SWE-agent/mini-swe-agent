import json
import time

from jinja2 import StrictUndefined, Template

from minisweagent.exceptions import FormatError
from minisweagent.models.litellm_model import LitellmModel, LitellmModelConfig

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


class LitellmToolcallModelConfig(LitellmModelConfig):
    format_error_template: str = "Unknown tool '{{tool_name}}'. Valid tools: {{valid_tools}}"


class LitellmToolcallModel(LitellmModel):
    def __init__(self, **kwargs):
        super().__init__(config_class=LitellmToolcallModelConfig, **kwargs)

    def _query(self, messages: list[dict[str, str]], **kwargs):
        return super()._query(messages, tools=[BASH_TOOL], **kwargs)

    def parse_actions(self, response) -> list[dict]:
        """Parse tool calls from the response. Raises FormatError if unknown tool."""
        tool_calls = response.choices[0].message.tool_calls or []
        actions = []
        for tool_call in tool_calls:
            if tool_call.function.name != "bash":
                raise FormatError(
                    {
                        "role": "user",
                        "content": f"Unknown tool '{tool_call.function.name}'. Valid tools: ['bash']",
                        "extra": {
                            "interrupt_type": "FormatError",
                            "tool_name": tool_call.function.name,
                            "valid_tools": ["bash"],
                        },
                    }
                )
            args = json.loads(tool_call.function.arguments)
            actions.append({"command": args["command"], "tool_call_id": tool_call.id})
        return actions

    def format_observation_messages(
        self, message: dict, outputs: list[dict], template_vars: dict | None = None
    ) -> list[dict]:
        """Format execution outputs into tool result messages."""
        results = []
        actions = message.get("extra", {}).get("actions", [])
        for action, output in zip(actions, outputs):
            content = Template(self.config.observation_template, undefined=StrictUndefined).render(
                output=output, **(template_vars or {})
            )
            results.append(
                self.format_message(
                    role="tool",
                    tool_call_id=action["tool_call_id"],
                    content=content,
                    extra={
                        "raw_output": output.get("output", ""),
                        "returncode": output.get("returncode"),
                        "timestamp": time.time(),
                        **(
                            {"exception_info": output["exception_info"]} | output.get("extra", {})
                            if output.get("exception_info")
                            else {}
                        ),
                    },
                )
            )
        return results

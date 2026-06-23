import logging
import time
from typing import Any

from pydantic import BaseModel

from minisweagent.models import GLOBAL_MODEL_STATS
from minisweagent.models.utils.actions_text import format_observation_messages
from minisweagent.models.utils.actions_toolcall import format_toolcall_observation_messages
from minisweagent.models.utils.actions_toolcall_response import (
    format_toolcall_observation_messages as format_response_api_observation_messages,
)
from minisweagent.models.utils.openai_multimodal import expand_multimodal_content


def make_output(content: str, actions: list[dict], cost: float = 1.0) -> dict:
    """Helper to create an output dict for DeterministicModel.

    Args:
        content: The response content string
        actions: List of action dicts, e.g., [{"command": "echo hello"}]
        cost: Cost to report for this output (default 1.0)
    """
    return {
        "role": "assistant",
        "content": content,
        "extra": {"actions": actions, "cost": cost, "timestamp": time.time()},
    }


def make_toolcall_output(content: str | None, tool_calls: list[dict], actions: list[dict]) -> dict:
    """Helper to create a toolcall output dict for DeterministicToolcallModel.

    Args:
        content: Optional text content (can be None for tool-only responses)
        tool_calls: List of tool call dicts in OpenAI format
        actions: List of parsed action dicts, e.g., [{"command": "echo hello", "tool_call_id": "call_123"}]
    """
    return {
        "role": "assistant",
        "content": content,
        "tool_calls": tool_calls,
        "extra": {"actions": actions, "cost": 1.0, "timestamp": time.time()},
    }


def make_response_api_output(content: str | None, actions: list[dict]) -> dict:
    """Helper to create an output dict for DeterministicResponseAPIToolcallModel.

    Args:
        content: Optional text content (can be None for tool-only responses)
        actions: List of action dicts with 'command' and 'tool_call_id' keys
    """
    output_items = []
    if content:
        output_items.append(
            {"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": content}]}
        )
    for action in actions:
        output_items.append(
            {
                "type": "function_call",
                "call_id": action["tool_call_id"],
                "name": "bash",
                "arguments": f'{{"command": "{action["command"]}"}}',
            }
        )
    return {
        "object": "response",
        "output": output_items,
        "extra": {"actions": actions, "cost": 1.0, "timestamp": time.time()},
    }


def _process_test_actions(actions: list[dict]) -> bool:
    """Process special test actions. Returns True if the query should be retried."""
    for action in actions:
        if "raise" in action:
            raise action["raise"]
        cmd = action.get("command", "")
        if cmd.startswith("/sleep "):
            time.sleep(float(cmd.split("/sleep ")[1]))
            return True
        if cmd.startswith("/warning"):
            logging.warning(cmd.split("/warning")[1])
            return True
    return False


class DeterministicModelConfig(BaseModel):
    outputs: list[dict]
    """List of exact output messages to return in sequence. Each dict should have 'role', 'content', and 'extra' (with 'actions')."""
    model_name: str = "deterministic"
    cost_per_call: float = 1.0
    observation_template: str = (
        "{% if output.exception_info %}<exception>{{output.exception_info}}</exception>\n{% endif %}"
        "<returncode>{{output.returncode}}</returncode>\n<output>\n{{output.output}}</output>"
    )
    """Template used to render the observation after executing an action."""
    multimodal_regex: str = ""
    """Regex to extract multimodal content. Empty string disables multimodal processing."""


class DeterministicModel:
    def __init__(self, **kwargs):
        """Initialize with a list of output messages to return in sequence."""
        self.config = DeterministicModelConfig(**kwargs)
        self.current_index = -1

    def query(self, messages: list[dict[str, str]], **kwargs) -> dict:
        self.current_index += 1
        output = self.config.outputs[self.current_index]
        if _process_test_actions(output.get("extra", {}).get("actions", [])):
            return self.query(messages, **kwargs)
        GLOBAL_MODEL_STATS.add(self.config.cost_per_call)
        return output

    def format_message(self, **kwargs) -> dict:
        return expand_multimodal_content(kwargs, pattern=self.config.multimodal_regex)

    def format_observation_messages(
        self, message: dict, outputs: list[dict], template_vars: dict | None = None
    ) -> list[dict]:
        """Format execution outputs into observation messages."""
        return format_observation_messages(
            outputs,
            observation_template=self.config.observation_template,
            template_vars=template_vars,
            multimodal_regex=self.config.multimodal_regex,
        )

    def get_template_vars(self, **kwargs) -> dict[str, Any]:
        return self.config.model_dump()

    def serialize(self) -> dict:
        return {
            "info": {
                "config": {
                    "model": self.config.model_dump(mode="json"),
                    "model_type": f"{self.__class__.__module__}.{self.__class__.__name__}",
                },
            }
        }


class DeterministicToolcallModelConfig(BaseModel):
    outputs: list[dict]
    """List of exact output messages with tool_calls to return in sequence."""
    model_name: str = "deterministic_toolcall"
    cost_per_call: float = 1.0
    observation_template: str = (
        "{% if output.exception_info %}<exception>{{output.exception_info}}</exception>\n{% endif %}"
        "<returncode>{{output.returncode}}</returncode>\n<output>\n{{output.output}}</output>"
    )
    """Template used to render the observation after executing an action."""
    multimodal_regex: str = ""
    """Regex to extract multimodal content. Empty string disables multimodal processing."""


class DeterministicToolcallModel:
    def __init__(self, **kwargs):
        """Initialize with a list of toolcall output messages to return in sequence."""
        self.config = DeterministicToolcallModelConfig(**kwargs)
        self.current_index = -1

    def query(self, messages: list[dict[str, str]], **kwargs) -> dict:
        self.current_index += 1
        output = self.config.outputs[self.current_index]
        if _process_test_actions(output.get("extra", {}).get("actions", [])):
            return self.query(messages, **kwargs)
        GLOBAL_MODEL_STATS.add(self.config.cost_per_call)
        return output

    def format_message(self, **kwargs) -> dict:
        return expand_multimodal_content(kwargs, pattern=self.config.multimodal_regex)

    def format_observation_messages(
        self, message: dict, outputs: list[dict], template_vars: dict | None = None
    ) -> list[dict]:
        """Format execution outputs into tool result messages."""
        actions = message.get("extra", {}).get("actions", [])
        return format_toolcall_observation_messages(
            actions=actions,
            outputs=outputs,
            observation_template=self.config.observation_template,
            template_vars=template_vars,
            multimodal_regex=self.config.multimodal_regex,
        )

    def get_template_vars(self, **kwargs) -> dict[str, Any]:
        return self.config.model_dump()

    def serialize(self) -> dict:
        return {
            "info": {
                "config": {
                    "model": self.config.model_dump(mode="json"),
                    "model_type": f"{self.__class__.__module__}.{self.__class__.__name__}",
                },
            }
        }


class DeterministicResponseAPIToolcallModelConfig(BaseModel):
    outputs: list[dict]
    """List of exact Response API output messages to return in sequence."""
    model_name: str = "deterministic_response_api_toolcall"
    cost_per_call: float = 1.0
    observation_template: str = (
        "{% if output.exception_info %}<exception>{{output.exception_info}}</exception>\n{% endif %}"
        "<returncode>{{output.returncode}}</returncode>\n<output>\n{{output.output}}</output>"
    )
    """Template used to render the observation after executing an action."""
    multimodal_regex: str = ""
    """Regex to extract multimodal content. Empty string disables multimodal processing."""


class DeterministicResponseAPIToolcallModel:
    """Deterministic test model using OpenAI Responses API format."""

    def __init__(self, **kwargs):
        """Initialize with a list of Response API output messages to return in sequence."""
        self.config = DeterministicResponseAPIToolcallModelConfig(**kwargs)
        self.current_index = -1

    def query(self, messages: list[dict[str, str]], **kwargs) -> dict:
        self.current_index += 1
        output = self.config.outputs[self.current_index]
        if _process_test_actions(output.get("extra", {}).get("actions", [])):
            return self.query(messages, **kwargs)
        GLOBAL_MODEL_STATS.add(self.config.cost_per_call)
        return output

    def format_message(self, **kwargs) -> dict:
        """Format message in Responses API format."""
        role = kwargs.get("role", "user")
        content = kwargs.get("content", "")
        extra = kwargs.get("extra")
        content_items = [{"type": "input_text", "text": content}] if isinstance(content, str) else content
        msg: dict = {"type": "message", "role": role, "content": content_items}
        if extra:
            msg["extra"] = extra
        return msg

    def format_observation_messages(
        self, message: dict, outputs: list[dict], template_vars: dict | None = None
    ) -> list[dict]:
        """Format execution outputs into function_call_output messages."""
        actions = message.get("extra", {}).get("actions", [])
        return format_response_api_observation_messages(
            actions=actions,
            outputs=outputs,
            observation_template=self.config.observation_template,
            template_vars=template_vars,
            multimodal_regex=self.config.multimodal_regex,
        )

    def get_template_vars(self, **kwargs) -> dict[str, Any]:
        return self.config.model_dump()

    def serialize(self) -> dict:
        return {
            "info": {
                "config": {
                    "model": self.config.model_dump(mode="json"),
                    "model_type": f"{self.__class__.__module__}.{self.__class__.__name__}",
                },
            }
        }

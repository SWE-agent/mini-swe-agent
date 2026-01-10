import logging
import re
import time
from dataclasses import dataclass
from typing import Any

from jinja2 import StrictUndefined, Template
from pydantic import BaseModel

from minisweagent.exceptions import FormatError
from minisweagent.models import GLOBAL_MODEL_STATS


@dataclass
class _MockMessage:
    content: str


@dataclass
class _MockChoice:
    message: _MockMessage


@dataclass
class _MockResponse:
    """Minimal response object for DeterministicModel."""

    choices: list[_MockChoice]


class DeterministicModelConfig(BaseModel):
    outputs: list[str]
    model_name: str = "deterministic"
    cost_per_call: float = 1.0
    action_regex: str = r"```mswea_bash_command\s*\n(.*?)\n```"
    """Regex to extract the action from the LM's output."""
    format_error_template: str = (
        "Please always provide EXACTLY ONE action in triple backticks, found {{actions|length}} actions."
    )
    """Template used when the LM's output is not in the expected format."""
    action_observation_template: str = (
        "{% if output.exception_info %}<exception>{{output.exception_info}}</exception>\n{% endif %}"
        "<returncode>{{output.returncode}}</returncode>\n<output>\n{{output.output}}</output>"
    )
    """Template used to render the observation after executing an action."""


class DeterministicModel:
    def __init__(self, **kwargs):
        """
        Initialize with a list of outputs to return in sequence.
        """
        self.config = DeterministicModelConfig(**kwargs)
        self.current_index = -1
        self.cost = 0.0
        self.n_calls = 0

    def query(self, messages: list[dict[str, str]], **kwargs) -> dict:
        self.current_index += 1
        output = self.config.outputs[self.current_index]
        if "/sleep" in output:
            print("SLEEPING")
            time.sleep(float(output.split("/sleep")[1]))
            return self.query(messages, **kwargs)
        if "/warning" in output:
            logging.warning(output.split("/warning")[1])
            return self.query(messages, **kwargs)
        cost_output = self._calculate_cost()
        self.n_calls += 1
        self.cost += cost_output["cost"]
        GLOBAL_MODEL_STATS.add(cost_output["cost"])
        response = _MockResponse(choices=[_MockChoice(message=_MockMessage(content=output))])
        return {
            "role": "assistant",
            "content": output,
            "extra": {
                "actions": self.parse_actions(response),
                **cost_output,
                "timestamp": time.time(),
            },
        }  # DeterministicModel doesn't have a real message object to preserve

    def _calculate_cost(self) -> dict[str, float]:
        return {"cost": self.config.cost_per_call}

    def parse_actions(self, response) -> list[dict]:
        """Parse actions from the model response. Raises FormatError if not exactly one action."""
        content = response.choices[0].message.content or ""
        actions = [a.strip() for a in re.findall(self.config.action_regex, content, re.DOTALL)]
        if len(actions) != 1:
            raise FormatError(
                {
                    "role": "user",
                    "content": Template(self.config.format_error_template, undefined=StrictUndefined).render(
                        actions=actions
                    ),
                    "extra": {
                        "interrupt_type": "FormatError",
                        "n_actions": len(actions),
                        "model_response": content,
                    },
                }
            )
        return [{"command": action} for action in actions]

    def format_actions_output(self, message: dict, outputs: list[dict]) -> list[dict]:
        """Format execution outputs into observation messages."""
        results = []
        for output in outputs:
            content = Template(self.config.action_observation_template, undefined=StrictUndefined).render(output=output)
            results.append(
                {
                    "role": "user",
                    "content": content,
                    "extra": {
                        "raw_output": output.get("output", ""),
                        "returncode": output.get("returncode"),
                        "timestamp": time.time(),
                        **(
                            {"exception_info": output["exception_info"]} | output.get("extra", {})
                            if output.get("exception_info")
                            else {}
                        ),
                    },
                }
            )
        return results

    def get_template_vars(self, **kwargs) -> dict[str, Any]:
        return self.config.model_dump() | {"n_model_calls": self.n_calls, "model_cost": self.cost}

    def serialize(self) -> dict:
        return {
            "info": {
                "model_stats": {
                    "instance_cost": self.cost,
                    "api_calls": self.n_calls,
                },
                "config": {
                    "model": self.config.model_dump(mode="json"),
                    "model_type": f"{self.__class__.__module__}.{self.__class__.__name__}",
                },
            }
        }

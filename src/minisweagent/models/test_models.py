import logging
import re
import time
from typing import Any

from jinja2 import StrictUndefined, Template
from pydantic import BaseModel

from minisweagent.exceptions import FormatError
from minisweagent.models import GLOBAL_MODEL_STATS


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


class DeterministicModel:
    def __init__(self, **kwargs):
        """
        Initialize with a list of outputs to return in sequence.
        """
        self.config = DeterministicModelConfig(**kwargs)
        self.current_index = -1
        self.cost = 0.0
        self.n_calls = 0

    def query(self, messages: list[dict[str, str]], **kwargs) -> list[dict]:
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
        return [
            {
                "role": "assistant",
                "content": output,
                "extra": {
                    "action": self.parse_action(output),
                    **cost_output,
                    "timestamp": time.time(),
                },
            }
        ]

    def _calculate_cost(self) -> dict[str, float]:
        return {"cost": self.config.cost_per_call}

    def parse_action(self, content: str) -> str:
        """Parse the action from the model output. Raises InterruptAgentFlow if not exactly one action."""
        actions = re.findall(self.config.action_regex, content, re.DOTALL)
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
        return actions[0].strip()

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

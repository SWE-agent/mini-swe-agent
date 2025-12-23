"""Basic agent class. See https://mini-swe-agent.com/latest/advanced/control_flow/ for visual explanation."""

import json
import traceback
from pathlib import Path

from jinja2 import StrictUndefined, Template
from pydantic import BaseModel

from minisweagent import Environment, Model, __version__
from minisweagent.exceptions import LimitsExceeded, NonTerminatingException, TerminatingException
from minisweagent.utils.serialize import recursive_merge


class AgentConfig(BaseModel):
    """Check the config files in minisweagent/config for example settings."""

    system_template: str
    """Template for the system message (the first message)."""
    instance_template: str
    """Template for the first user message specifying the task (the second message overall)."""
    step_limit: int = 0
    """Maximum number of steps the agent can take."""
    cost_limit: float = 3.0
    """Stop agent after exceeding (!) this cost."""
    output_path: Path | None = None
    """Save the trajectory to this path."""


class DefaultAgent:
    def __init__(self, model: Model, env: Environment, *, config_class: type = AgentConfig, **kwargs):
        """See the `AgentConfig` class for permitted keyword arguments."""
        self.config = config_class(**kwargs)
        self.messages: list[dict] = []
        self.model = model
        self.env = env
        self.extra_template_vars = {}

    def render_template(self, template: str, **kwargs) -> str:
        template_vars = self.config.model_dump() | self.env.get_template_vars()
        return Template(template, undefined=StrictUndefined).render(
            **kwargs, **template_vars, **self.extra_template_vars
        )

    def add_messages(self, messages: list[dict]):
        self.messages.extend(messages)

    def run(self, task: str, **kwargs) -> dict:
        """Run step() until agent is finished. Returns dictionary with exit_status, submission keys."""
        self.extra_template_vars |= {"task": task, **kwargs}
        self.messages = []
        self.add_messages([{"role": "system", "content": self.render_template(self.config.system_template)}])
        self.add_messages([{"role": "user", "content": self.render_template(self.config.instance_template)}])
        while True:
            info = {}
            try:
                self.step()
            except Exception as e:
                self.add_messages([{"role": "user", "content": str(e)}])
                if isinstance(e, NonTerminatingException):
                    continue
                info = {"exit_status": type(e).__name__, "submission": str(e)}
                if isinstance(e, TerminatingException):
                    return info
                info["traceback"] = traceback.format_exc()
                raise e
            finally:
                self.save(self.config.output_path, {"info": info})

    def step(self):
        """Query the LM, execute actions."""
        self.execute_actions(self.query())

    def query(self) -> list[dict]:
        """Query the model and return model messages. Override to add hooks."""
        if 0 < self.config.step_limit <= self.model.n_calls or 0 < self.config.cost_limit <= self.model.cost:
            raise LimitsExceeded()
        messages = self.model.query(self.messages)
        self.add_messages(messages)
        return messages

    def execute_actions(self, messages: list[dict]) -> list[dict]:
        """Execute actions in messages, add all messages, return observation messages. Override to add hooks."""
        observation_messages = self.env.execute_messages(messages)
        self.add_messages(observation_messages)
        return observation_messages

    def serialize(self) -> dict:
        """Serialize agent state to a json-compatible nested dictionary for saving."""
        return {
            "info": {
                "config": {
                    "agent": self.config.model_dump(mode="json"),
                    "agent_type": f"{self.__class__.__module__}.{self.__class__.__name__}",
                },
            },
            "messages": self.messages,
        }

    def save(self, path: Path | None, *extra_dicts) -> dict:
        """Save the trajectory of the agent to a file if path is given. Returns full serialized data.
        You can pass additional dictionaries with extra data to be (recursively) merged into the output data.
        """
        data = recursive_merge(self.serialize(), self.model.serialize(), self.env.serialize(), *extra_dicts)
        data["trajectory_format"] = "mini-swe-agent-1"
        data["info"]["mini_version"] = __version__
        if path:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(data, indent=2))
        return data

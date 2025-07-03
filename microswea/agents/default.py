import re
import subprocess
from dataclasses import dataclass
from typing import Callable

from jinja2 import Template
from rich.console import Console

from microswea import Environment, Model


@dataclass
class AgentConfig:
    system_template: str
    instance_template: str
    step_limit: int = 0
    cost_limit: float = 3.0


console = Console(highlight=False)  # Print with colors


class NonTerminatingException(Exception):
    """Raising this exception will add an error message to the agent history."""


class TerminatingException(Exception):
    """Raising this exception will terminate the agent."""


class DefaultAgent:
    def __init__(
        self, model: Model, env: Environment, problem_statement: str, config_class: Callable = AgentConfig, **kwargs
    ):
        self.config = config_class(**kwargs)
        self.messages = []
        self.model = model
        self.env = env
        self.add_message("system", self.config.system_template)
        self.add_message("user", Template(self.config.instance_template).render(problem_statement=problem_statement))

    def add_message(self, role: str, content: str):
        self.messages.append({"role": role, "content": content})
        color = "red" if role == "user" else "green"
        console.print(f"\n[bold {color}]{role}:[/bold {color}]\n{content}", highlight=False)

    def run(self) -> str:
        """Run step() until agent is finished. Return final observation."""
        while True:
            try:
                self.step()
            except NonTerminatingException as e:
                self.add_message("user", str(e))
            except TerminatingException as e:
                self.add_message("user", str(e))
                return str(e)

    def step(self) -> str:
        """Query the LM, execute the action, return the observation."""
        if 0 < self.config.step_limit <= self.model.n_calls or 0 < self.config.cost_limit <= self.model.cost:
            raise TerminatingException("limits_exceeded")

        message = self.query()
        self.add_message("assistant", message)
        observation = self.get_observation(message)
        self.add_message("user", observation)
        return observation

    def query(self) -> str:
        """Query the model and return the response."""
        return self.model.query(self.messages)

    def get_observation(self, message: str) -> str:
        """Execute the action and return the observation."""
        action = self.parse_action(message)
        return self.execute_action(action)

    def parse_action(self, message: str) -> str:
        """Parse the action from the message. Returns the action."""
        actions = re.findall(r"```[a-zA-Z]*\n(.*?)(?=\n```|```)", message, re.DOTALL)
        if len(actions) == 1:
            return actions[0].strip()
        raise NonTerminatingException("Please always provide EXACTLY ONE action in triple backticks.")

    def execute_action(self, action: str) -> str:
        try:
            output = self.env.execute(action)
        except (TimeoutError, subprocess.TimeoutExpired):
            raise NonTerminatingException(
                "The command timed out. Please change your command and make sure it doesn't require input."
            )
        self.has_finished(output)
        return "\n".join([f"<{key}>\n{value}\n</{key}>" for key, value in output.items()])

    def has_finished(self, output: dict[str, str]):
        """Raises TerminatingException with final output if the agent has finished its task."""
        if output.get("stdout") and output["stdout"].splitlines()[0] == "NANO_SWE_AGENT_FINAL_OUTPUT":
            raise TerminatingException("\n".join(output["stdout"].splitlines()[1:]) or "EMPTY_OUTPUT")

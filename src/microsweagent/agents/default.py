import re
import subprocess
from collections.abc import Callable
from dataclasses import dataclass

from jinja2 import Template

from microsweagent import Environment, Model


@dataclass
class AgentConfig:
    # The default settings are the bare minimum to run the agent. Take a look at the config files for improved settings.
    system_template: str = "You are a helpful assistant that can do anything."
    instance_template: str = (
        "Your task: {{task}}. Please reply with a single shell command in triple backticks. "
        "To finish, the first line of the output of the shell command must be 'MICRO_SWE_AGENT_FINAL_OUTPUT'."
    )
    timeout_template: str = "The command timed out. Please change your command and make sure it doesn't require input."
    format_error_template: str = "Please always provide EXACTLY ONE action in triple backticks."
    action_observation_template: str = "Observation: {{output}}"
    step_limit: int = 0
    cost_limit: float = 3.0


class NonTerminatingException(Exception):
    """Raised for conditions that can be handled by the agent."""


class FormatError(NonTerminatingException):
    """Raised when the LM's output is not in the expected format."""


class ExecutionTimeoutError(NonTerminatingException):
    """Raised when the action execution timed out."""


class TerminatingException(Exception):
    """Raised for conditions that terminate the agent."""


class Submitted(TerminatingException):
    """Raised when the LM declares that the agent has finished its task."""


class LimitsExceeded(TerminatingException):
    """Raised when the agent has reached its cost or step limit."""


class DefaultAgent:
    def __init__(self, model: Model, env: Environment, config_class: Callable = AgentConfig, **kwargs):
        self.config = config_class(**kwargs)
        self.messages: list[dict] = []
        self.model = model
        self.env = env

    def add_message(self, role: str, content: str):
        self.messages.append({"role": role, "content": content})

    def run(self, task: str) -> tuple[str, str]:
        """Run step() until agent is finished. Return exit status & message"""
        self.messages = []
        self.add_message("system", self.config.system_template)
        self.add_message("user", Template(self.config.instance_template).render(task=task))
        while True:
            try:
                self.step()
            except NonTerminatingException as e:
                self.add_message("user", str(e))
            except TerminatingException as e:
                self.add_message("user", str(e))
                return type(e).__name__, str(e)

    def step(self) -> str:
        """Query the LM, execute the action, return the observation."""
        message = self.query()
        self.add_message("assistant", message)
        observation = self.get_observation(message)
        self.add_message("user", observation)
        return observation

    def query(self) -> str:
        """Query the model and return the response."""
        if 0 < self.config.step_limit <= self.model.n_calls or 0 < self.config.cost_limit <= self.model.cost:
            raise LimitsExceeded()

        return self.model.query(self.messages)

    def get_observation(self, message: str) -> str:
        """Execute the action and return the observation."""
        return self.execute_action(self.parse_action(message))

    def parse_action(self, message: str) -> str:
        """Parse the action from the message. Returns the action."""
        actions = re.findall(r"```[a-zA-Z]*\n(.*?)(?=\n```|```)", message, re.DOTALL)
        if len(actions) == 1:
            return actions[0].strip()
        raise FormatError(Template(self.config.format_error_template).render(actions=actions))

    def execute_action(self, action: str) -> str:
        try:
            output = self.env.execute(action)
        except (TimeoutError, subprocess.TimeoutExpired):
            raise ExecutionTimeoutError(Template(self.config.timeout_template).render(action=action))
        self.has_finished(output)
        return Template(self.config.action_observation_template).render(output=output)

    def has_finished(self, output: dict[str, str]):
        """Raises Submitted exception with final output if the agent has finished its task."""
        if output.get("stdout") and output["stdout"].splitlines()[0] == "MICRO_SWE_AGENT_FINAL_OUTPUT":
            raise Submitted("\n".join(output["stdout"].splitlines()[1:]))

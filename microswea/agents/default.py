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


class DefaultAgent:
    def __init__(
        self, model: Model, env: Environment, problem_statement: str, config_class: Callable = AgentConfig, **kwargs
    ):
        self.config = config_class(**kwargs)
        instance_message = Template(self.config.instance_template).render(problem_statement=problem_statement)
        console.print(f"[bold green]System template:[/bold green]\n{self.config.system_template}", highlight=False)
        console.print(f"[bold green]Instance message:[/bold green]\n{instance_message}", highlight=False)
        self.messages = [
            {"role": "system", "content": self.config.system_template},
            {"role": "user", "content": instance_message},
        ]
        self.model = model
        self.env = env

    def run(self) -> str:
        """Run the agent and return the final observation.
        Essentially just calls `step` until the agent is finished.
        """
        is_finished = False
        while not is_finished:
            is_finished, observation = self.step()
        return observation

    def step(self) -> tuple[bool, str]:
        """Query the LM and execute the action

        Returns:
            is_finished: whether the agent has finished its task
            observation: The observation or error message
        """
        if 0 < self.config.step_limit <= self.model.n_calls or 0 < self.config.cost_limit <= self.model.cost:
            return True, "limits_exceeded"

        return self.get_observation(self.query())

    def query(self) -> str:
        """Query the model and return the response."""
        message = self.model.query(self.messages)
        console.print(f"[bold red]Assistant (step {self.model.n_calls}, ${self.model.cost:.2f}):[/bold red]\n{message}")
        self.messages.append({"role": "assistant", "content": message})
        return message

    def get_observation(self, message: str) -> tuple[bool, str]:
        """Execute the action and return the observation.
        If the first line of the stdout of the action is "NANO_SWE_AGENT_FINAL_OUTPUT", assume the agent has finished its task.

        Returns:
            is_finished: whether the agent has finished its task
            observation: The observation or error message
        """
        error, action = self.parse_action(message)
        if error:
            return False, error
        is_finished, observation = self.execute_action(action)
        self.messages.append({"role": "user", "content": observation})
        console.print(f"[bold green]Observation (step {self.model.n_calls}):[/bold green]\n{observation}")
        return is_finished, observation

    def parse_action(self, message: str) -> tuple[str, str]:
        """Parse the action from the message. Returns an optional error message and the action."""
        actions = re.findall(r"```[a-zA-Z]*\n(.*?)(?=\n```|```)", message, re.DOTALL)
        if len(actions) == 1:
            return "", actions[0].strip()
        return "Please always provide EXACTLY ONE action in triple backticks.", ""

    def execute_action(self, action: str) -> tuple[bool, str]:
        try:
            output = self.env.execute(action)
        except (TimeoutError, subprocess.TimeoutExpired):
            return False, "The command timed out. Please change your command and make sure it doesn't require input."
        except Exception as e:
            return False, f"Error executing action: {e}"
        if final_output := self.get_final_output(output):
            return True, final_output
        return False, "\n".join([f"<{key}>\n{value}\n</{key}>" for key, value in output.items()])

    def get_final_output(self, output: dict[str, str]) -> str:
        """Check whether the agent has finished its task. Returning a non-empty string as final output will terminate the agent."""
        if output.get("stdout") and output["stdout"].splitlines()[0] == "NANO_SWE_AGENT_FINAL_OUTPUT":
            return "\n".join(output["stdout"].splitlines()[1:]) or "EMPTY_OUTPUT"
        return ""

import re
import subprocess
from dataclasses import dataclass

from jinja2 import Template
from rich.console import Console
from rich.prompt import Prompt

from nanoswea import Environment, Model


@dataclass
class AgentConfig:
    system_template: str
    instance_template: str
    step_limit: int = 0
    cost_limit: float = 3.0
    confirm_actions: bool = True

console = Console(highlight=False)  # Print with colors


class Agent:
    def __init__(self, config: AgentConfig, model: Model, env: Environment, problem_statement: str):
        self.config = config
        instance_message = Template(config.instance_template).render(problem_statement=problem_statement)
        console.print(f"[bold green]System template:[/bold green]\n{config.system_template}", highlight=False)
        console.print(f"[bold green]Instance message:[/bold green]\n{instance_message}", highlight=False)
        self.history = [
            {"role": "system", "content": config.system_template},
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
            is_finished, _, observation = self.step()
        return observation

    def step(self) -> tuple[bool, str, str]:
        """Query the LM and execute the action

        Returns:
            is_finished: whether the agent has finished its task
            response: Model reponse (if any)
            observation: The observation or error message
        """
        if 0 < self.config.step_limit <= self.model.n_calls or 0 < self.config.cost_limit <= self.model.cost:
            return True, "", "limits_exceeded"

        message = self.model.query(self.history)
        console.print(f"[bold red]Assistant (step {self.model.n_calls}, ${self.model.cost:.2f}):[/bold red]\n{message}")
        self.history.append({"role": "assistant", "content": message})

        error, action = self.parse_action(message)
        is_finished, observation = self.get_observation(error, action)
        self.history.append({"role": "user", "content": observation})
        console.print(f"[bold green]Observation (step {self.model.n_calls}):[/bold green]\n{observation}")

        return is_finished, message, observation

    def parse_action(self, message: str) -> tuple[str, str]:
        """Parse the action from the message. Returns an optional error message and the action."""
        actions = re.findall(r"```[a-zA-Z]*\n(.*?)(?=\n```|```)", message, re.DOTALL)
        if len(actions) == 1:
            return "", actions[0].strip()
        return "Please always provide EXACTLY ONE action in triple backticks.", ""

    def get_observation(self, error: str, action: str) -> tuple[bool, str]:
        """Execute the action and return the observation.
        If the first line of the stdout of the action is "NANO_SWE_AGENT_FINAL_OUTPUT", assume the agent has finished its task.

        Returns:
            is_finished: whether the agent has finished its task
            observation: The observation or error message
        """
        if error:
            return False, error
        if rejection_message := self.reject_action(action):
            return False, rejection_message
        try:
            output = self.env.execute(action)
        except (TimeoutError, subprocess.TimeoutExpired):
            return False, "The command timed out. Please change your command and make sure it doesn't require input."
        except Exception as e:
            return False, f"Error executing action: {e}"
        if final_output := self.get_final_output(output):
            return True, final_output
        return False, "\n".join([f"<{key}>\n{value}\n</{key}>" for key, value in output.items()])

    def reject_action(self, action: str) -> str:
        """Ask the user to confirm the action. Returns a rejection message if the user rejects the action."""
        if self.config.confirm_actions:
            if response := Prompt.ask(
                "[bold yellow]Execute?[/bold yellow] ([green][bold]Enter[/bold] to confirm[/green], or enter rejection message)"
            ):
                return f"Command not executed. The user rejected your command with the following message: {response}"
        return ""

    def get_final_output(self, output: dict[str, str]) -> str:
        """Check whether the agent has finished its task. Returning a non-empty string as final output will terminate the agent."""
        if output.get("stdout") and output["stdout"].splitlines()[0] == "NANO_SWE_AGENT_FINAL_OUTPUT":
            return "\n".join(output["stdout"].splitlines()[1:]) or "EMPTY_OUTPUT"
        return ""
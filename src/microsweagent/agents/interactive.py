"""A small generalization of the default agent that puts the user in the loop."""

import re
from dataclasses import dataclass, field
from typing import Literal

from rich.console import Console

from microsweagent.agents.default import AgentConfig, DefaultAgent, NonTerminatingException

console = Console(highlight=False)


@dataclass
class InteractiveAgentConfig(AgentConfig):
    mode: Literal["human", "confirm", "yolo"] = "confirm"
    """Whether to confirm actions."""
    whitelist_actions: list[str] = field(default_factory=list)
    """Never confirm actions that match these regular expressions."""


class InteractiveAgent(DefaultAgent):
    _MODE_COMMANDS_MAPPING = {"/u": "human", "/c": "confirm", "/y": "yolo"}

    def __init__(self, *args, **kwargs):
        super().__init__(*args, config_class=InteractiveAgentConfig, **kwargs)
        self.cost_last_confirmed = 0.0

    def add_message(self, role: str, content: str):
        super().add_message(role, content)
        if role == "assistant":
            console.print(
                f"\n[red][bold]micro-swe-agent[/bold] (step [bold]{self.model.n_calls}[/bold], [bold]${self.model.cost:.2f}[/bold]):[/red]\n",
                end="",
                highlight=False,
            )
        else:
            console.print(f"\n[bold green]{role.capitalize()}[/bold green]:\n", end="", highlight=False)
        console.print(content, highlight=False, markup=False)

    def query(self) -> str:
        if self.config.mode == "human":
            try:
                command = console.input(
                    "(enter command or hit [bold red]^C[/bold red] for menu) [bold yellow]>[/bold yellow] "
                )
            except KeyboardInterrupt:
                self.menu("asdf")
            return f"\n```bash\n{command}\n```"
        return super().query()

    def step(self) -> str:
        # Override the step method to handle user interruption
        try:
            return super().step()
        except KeyboardInterrupt:
            user_input = self.menu(
                "\n\n[bold yellow]Interrupted.[/bold yellow] "
                "[bold green]/h[/bold green] to show help, or [green]continue with comment/command[/green]"
                "\n[bold yellow]>[/bold yellow] "
            )
            if user_input:
                raise NonTerminatingException(f"Interrupted by user: {user_input}")
            raise NonTerminatingException(
                "Temporary interruption caught. Some actions may have been only partially executed."
            )

    def execute_action(self, action: str) -> str:
        # Override the execute_action method to handle user confirmation
        if self.config.mode == "confirm" and not any(re.match(r, action) for r in self.config.whitelist_actions):
            user_input = self.menu(
                "[bold yellow]Execute?[/bold yellow] [green][bold]Enter[/bold] to confirm[/green], "
                "[green bold]/h[/green bold] for help, "
                "or [green]enter comment/command[/green]\n"
                "[bold yellow]>[/bold yellow] "
            )
            match user_input.strip():
                case "" | "/y":
                    pass  # confirmed
                case "/u":
                    raise NonTerminatingException("Command not executed. Switching to human mode")
                case _:
                    raise NonTerminatingException(
                        f"Command not executed. The user rejected your command with the following message: {user_input}"
                    )
        return super().execute_action(action)

    def menu(self, prompt: str) -> str:
        """Prompts the user, takes care of /h (followed by requery) and sets the mode. Returns the user input."""
        user_input = console.input(prompt).strip()
        if user_input == "/h":
            console.print(
                f"Current mode: [bold green]{self.config.mode}[/bold green]\n"
                f"[bold green]/u[/bold green] to switch to human mode\n"
                f"[bold green]/c[/bold green] to switch to confirmation mode\n"
                f"[bold green]/y[/bold green] to switch to yolo mode\n"
            )
            return self.menu(prompt)
        if user_input in self._MODE_COMMANDS_MAPPING:
            if self.config.mode == self._MODE_COMMANDS_MAPPING[user_input]:
                return self.menu(f"[bold red]Already in {self.config.mode} mode.[/bold red]\n{prompt}")
            self.config.mode = self._MODE_COMMANDS_MAPPING[user_input]
            return user_input
        return user_input

"""A small generalization of the default agent that puts the user in the loop.

There are three modes:
- human: commands issued by the user are executed immediately
- confirm: commands issued by the LM but not whitelisted are confirmed by the user
- yolo: commands issued by the LM are executed immediately without confirmation
"""

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
        # Extend supermethod to print messages
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

    def query(self) -> dict:
        # Extend supermethod to handle human mode
        if self.config.mode == "human":
            match command := self._prompt_and_handle_special("[bold yellow]>[/bold yellow] "):
                case "/y" | "/c":  # Just go to the super query, which queries the LM for the next action
                    pass
                case _:
                    return {"content": f"\n```bash\n{command}\n```"}
        return super().query()

    def step(self) -> dict:
        # Override the step method to handle user interruption
        try:
            return super().step()
        except KeyboardInterrupt:
            # We always add a message about the interrupt and then just proceed to the next step
            interruption_message = self._prompt_and_handle_special(
                "\n\n[bold yellow]Interrupted.[/bold yellow] "
                "[bold green]/h[/bold green] to show help, or [green]continue with comment/command[/green]"
                "\n[bold yellow]>[/bold yellow] "
            ).strip()
            if not interruption_message or interruption_message in self._MODE_COMMANDS_MAPPING:
                interruption_message = "Temporary interruption caught."
            raise NonTerminatingException(f"Interrupted by user: {interruption_message}")

    def execute_action(self, action: dict) -> dict:
        # Override the execute_action method to handle user confirmation
        if self.should_ask_confirmation(action["action"]):
            self.ask_confirmation()
        return super().execute_action(action)

    def should_ask_confirmation(self, action: str) -> bool:
        return self.config.mode == "confirm" and not any(re.match(r, action) for r in self.config.whitelist_actions)

    def ask_confirmation(self) -> None:
        prompt = (
            "[bold yellow]Execute?[/bold yellow] [green][bold]Enter[/bold] to confirm[/green], "
            "[green bold]/h[/green bold] for help, "
            "or [green]enter comment/command[/green]\n"
            "[bold yellow]>[/bold yellow] "
        )
        match user_input := self._prompt_and_handle_special(prompt).strip():
            case "" | "/y":
                pass  # confirmed, do nothing
            case "/u":  # Skip execution action and get back to query
                raise NonTerminatingException("Command not executed. Switching to human mode")
            case _:
                raise NonTerminatingException(
                    f"Command not executed. The user rejected your command with the following message: {user_input}"
                )

    def _prompt_and_handle_special(self, prompt: str) -> str:
        """Prompts the user, takes care of /h (followed by requery) and sets the mode. Returns the user input."""
        user_input = console.input(prompt).strip()
        if user_input == "/h":
            console.print(
                f"Current mode: [bold green]{self.config.mode}[/bold green]\n"
                f"[bold green]/u[/bold green] to switch to human mode\n"
                f"[bold green]/c[/bold green] to switch to confirmation mode\n"
                f"[bold green]/y[/bold green] to switch to yolo mode\n"
            )
            return self._prompt_and_handle_special(prompt)
        if user_input in self._MODE_COMMANDS_MAPPING:
            if self.config.mode == self._MODE_COMMANDS_MAPPING[user_input]:
                return self._prompt_and_handle_special(
                    f"[bold red]Already in {self.config.mode} mode.[/bold red]\n{prompt}"
                )
            self.config.mode = self._MODE_COMMANDS_MAPPING[user_input]
            console.print(f"Switched to [bold green]{self.config.mode}[/bold green] mode.")
            return user_input
        return user_input

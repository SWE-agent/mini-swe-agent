"""A small generalization of the default agent that puts the user in the loop."""

import re
from dataclasses import dataclass, field

from rich.console import Console

from microswea.agents.default import AgentConfig, DefaultAgent, NonTerminatingException

console = Console(highlight=False)


@dataclass
class InteractiveAgentConfig(AgentConfig):
    confirm_actions: bool = True
    """Whether to confirm actions."""
    whitelist_actions: list[str] = field(default_factory=list)
    """Never confirm actions that match these regular expressions."""


class InteractiveAgent(DefaultAgent):
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

    def step(self) -> str:
        # Override the step method to handle user interruption
        try:
            return super().step()
        except KeyboardInterrupt:
            user_input = self.get_response(
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
        if self.config.confirm_actions and not any(re.match(r, action) for r in self.config.whitelist_actions):
            if user_input := self.get_response(
                "[bold yellow]Execute?[/bold yellow] [green][bold]Enter[/bold] to confirm[/green], "
                "[green bold]/h[/green bold] for help, "
                "or [green]enter comment/command[/green]\n"
                "[bold yellow]>[/bold yellow] "
            ):
                raise NonTerminatingException(
                    f"Command not executed. The user rejected your command with the following message: {user_input}"
                )
        return super().execute_action(action)

    def get_response(self, prompt: str) -> str:
        user_input = console.input(prompt)
        if user_input.strip() == "/h":
            console.print(
                "[bold green]/y[/bold green] to enter yolo mode (no need to confirm actions),\n"
                "[bold green]/x[/bold green] to exit yolo mode"
            )
            return self.get_response("[bold yellow]>[/bold yellow] ")
        if user_input.strip() == "/y":
            self.config.confirm_actions = False
            console.print("Yolo mode [bold red]enabled[/bold red].")
            return self.get_response(prompt)
        if user_input.strip() == "/x":
            self.config.confirm_actions = True
            console.print("Yolo mode [bold green]disabled[/bold green].")
            return self.get_response(prompt)
        return user_input

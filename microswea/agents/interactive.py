"""A small generalization of the default agent that allows for the
user in the loop.
"""

from dataclasses import dataclass

from rich.prompt import Prompt

from microswea.agents.default import AgentConfig, DefaultAgent, NonTerminatingException


@dataclass
class InteractiveAgentConfig(AgentConfig):
    confirm_actions: bool = True


class InteractiveAgent(DefaultAgent):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, config_class=InteractiveAgentConfig, **kwargs)

    def step(self) -> str:
        # Override the step method to handle user interruption
        try:
            return super().step()
        except KeyboardInterrupt:
            message = Prompt.ask("[bold red]Interrupted. Do you want to pass on a message?[/bold red]")
            raise NonTerminatingException(f"Interrupted by user: {message}")

    def execute_action(self, action: str) -> str:
        # Override the execute_action method to handle user confirmation
        if self.config.confirm_actions:
            if response := Prompt.ask(
                "[bold yellow]Execute?[/bold yellow] ([green][bold]Enter[/bold] to confirm[/green], or enter rejection message)"
            ):
                raise NonTerminatingException(
                    f"Command not executed. The user rejected your command with the following message: {response}"
                )
        return super().execute_action(action)

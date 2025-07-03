"""A small generalization of the default agent that allows for the
user in the loop.
"""

from dataclasses import dataclass

from rich.prompt import Prompt

from microswea.agents.default import AgentConfig, DefaultAgent


@dataclass
class InteractiveAgentConfig(AgentConfig):
    confirm_actions: bool = True


class InteractiveAgent(DefaultAgent):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, config_class=InteractiveAgentConfig, **kwargs)

    def run(self) -> str:
        # Override the run method to handle ^C
        is_finished = False
        while not is_finished:
            try:
                is_finished, observation = self.step()
            except KeyboardInterrupt:
                message = Prompt.ask("[bold red]Interrupted. Do you want to pass on a message?[/bold red]")
                self.messages.append({"role": "user", "content": message})
        return observation

    def execute_action(self, action: str) -> tuple[bool, str]:
        # Add a check for user confirmation
        if rejection_message := self.reject_action(action):
            return False, rejection_message
        return super().execute_action(action)

    def reject_action(self, action: str) -> str:
        """Ask the user to confirm the action. Returns a rejection message if the user rejects the action."""
        if self.config.confirm_actions:
            if response := Prompt.ask(
                "[bold yellow]Execute?[/bold yellow] ([green][bold]Enter[/bold] to confirm[/green], or enter rejection message)"
            ):
                return f"Command not executed. The user rejected your command with the following message: {response}"
        return ""

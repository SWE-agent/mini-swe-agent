"""A small generalization of the default agent that puts the user in the loop.

There are three modes:
- human: commands issued by the user are executed immediately
- confirm: commands issued by the LM but not whitelisted are confirmed by the user
- yolo: commands issued by the LM are executed immediately without confirmation
"""

import re
from typing import Literal, NoReturn

from prompt_toolkit.history import FileHistory
from prompt_toolkit.shortcuts import PromptSession
from rich.console import Console
from rich.rule import Rule

from minisweagent import global_config_dir
from minisweagent.agents.default import AgentConfig, DefaultAgent
from minisweagent.exceptions import LimitsExceeded, Submitted, UserInterruption
from minisweagent.models.utils.content_string import coerce_message_content

console = Console(highlight=False)
prompt_session = PromptSession(history=FileHistory(global_config_dir / "interactive_history.txt"))


class InteractiveAgentConfig(AgentConfig):
    mode: Literal["human", "confirm", "yolo"] = "confirm"
    """Whether to confirm actions."""
    whitelist_actions: list[str] = []
    """Never confirm actions that match these regular expressions."""
    confirm_exit: bool = True
    """If the agent wants to finish, do we ask for confirmation from user?"""


class InteractiveAgent(DefaultAgent):
    _MODE_COMMANDS_MAPPING = {"/u": "human", "/c": "confirm", "/y": "yolo"}

    def __init__(self, *args, config_class=InteractiveAgentConfig, **kwargs):
        super().__init__(*args, config_class=config_class, **kwargs)
        self.cost_last_confirmed = 0.0

    def add_messages(self, *messages: dict) -> list[dict]:
        # Extend supermethod to print messages
        for msg in messages:
            role, content = msg.get("role", "unknown"), coerce_message_content(msg)
            if role == "assistant":
                console.print(
                    f"\n[red][bold]mini-swe-agent[/bold] (step [bold]{self.n_calls}[/bold], [bold]${self.cost:.2f}[/bold]):[/red]\n",
                    end="",
                    highlight=False,
                )
            else:
                console.print(f"\n[bold green]{role.capitalize()}[/bold green]:\n", end="", highlight=False)
            console.print(content, highlight=False, markup=False)
        return super().add_messages(*messages)

    def query(self) -> dict:
        # Extend supermethod to handle human mode
        if self.config.mode == "human":
            match command := self._prompt_and_handle_slash_commands("[bold yellow]>[/bold yellow] "):
                case "/y" | "/c":
                    pass
                case _:
                    msg = {
                        "role": "assistant",
                        "content": f"\n```mswea_bash_command\n{command}\n```",
                        "extra": {"actions": [{"command": command}]},
                    }
                    self.add_messages(msg)
                    return msg
        try:
            with console.status("Waiting for the LM to respond..."):
                return super().query()
        except LimitsExceeded:
            console.print(
                f"Limits exceeded. Limits: {self.config.step_limit} steps, ${self.config.cost_limit}.\n"
                f"Current spend: {self.n_calls} steps, ${self.cost:.2f}."
            )
            self.config.step_limit = int(input("New step limit: "))
            self.config.cost_limit = float(input("New cost limit: "))
            return super().query()

    def step(self) -> list[dict]:
        # Override the step method to handle user interruption
        try:
            console.print(Rule())
            return super().step()
        except KeyboardInterrupt:
            interruption_message = self._prompt_and_handle_slash_commands(
                "\n\n[bold yellow]Interrupted.[/bold yellow] "
                "[green]Type a comment/command[/green] (/h for available commands)"
                "\n[bold yellow]>[/bold yellow] "
            ).strip()
            if not interruption_message or interruption_message in self._MODE_COMMANDS_MAPPING:
                interruption_message = "Temporary interruption caught."
            raise UserInterruption(
                {
                    "role": "user",
                    "content": f"Interrupted by user: {interruption_message}",
                    "extra": {"interrupt_type": "UserInterruption"},
                }
            )

    def execute_actions(self, message: dict) -> list[dict]:
        # Override to handle user confirmation and confirm_exit, with try/finally to preserve partial outputs
        actions = message.get("extra", {}).get("actions", [])
        outputs = []
        try:
            for action in actions:
                self._ask_confirmation_or_interrupt(action["command"])
                outputs.append(self.env.execute(action))
        except Submitted as e:
            self._check_for_new_task_or_submit(e)
        finally:
            result = self.add_messages(
                *self.model.format_observation_messages(message, outputs, self.get_template_vars())
            )
        return result

    def _check_for_new_task_or_submit(self, e: Submitted) -> NoReturn:
        """Check if user wants to add a new task or submit."""
        if self.config.confirm_exit:
            console.print(
                "[bold green]Agent wants to finish.[/bold green] "
                "[green]Type a comment to give it a new task or press enter to quit.\n"
                "[bold yellow]>[/bold yellow] ",
                end="",
            )
            if new_task := self._prompt_and_handle_slash_commands("").strip():
                raise UserInterruption(
                    {
                        "role": "user",
                        "content": f"The user added a new task: {new_task}",
                        "extra": {"interrupt_type": "UserNewTask"},
                    }
                )
        raise e

    def _should_ask_confirmation(self, action: str) -> bool:
        return self.config.mode == "confirm" and not any(re.match(r, action) for r in self.config.whitelist_actions)

    def _ask_confirmation_or_interrupt(self, action: str) -> None:
        if not self._should_ask_confirmation(action):
            return
        prompt = (
            f"[bold yellow]Action: [/bold yellow][blue]{action}[/blue]\n"
            "[bold yellow]Execute?[/bold yellow] [green][bold]Enter[/bold] to confirm[/green], "
            "or [green]Type a comment/command[/green] (/h for available commands)\n"
            "[bold yellow]>[/bold yellow] "
        )
        match user_input := self._prompt_and_handle_slash_commands(prompt).strip():
            case "" | "/y":
                pass  # confirmed, do nothing
            case "/u":  # Skip execution action and get back to query
                raise UserInterruption(
                    {
                        "role": "user",
                        "content": "Command not executed. Switching to human mode",
                        "extra": {"interrupt_type": "UserRejection"},
                    }
                )
            case _:
                raise UserInterruption(
                    {
                        "role": "user",
                        "content": f"Command not executed. The user rejected your command with the following message: {user_input}",
                        "extra": {"interrupt_type": "UserRejection"},
                    }
                )

    def _prompt_and_handle_slash_commands(self, prompt: str) -> str:
        """Prompts the user, takes care of /h (followed by requery) and sets the mode. Returns the user input."""
        console.print(prompt, end="")
        user_input = prompt_session.prompt("")
        if user_input == "/h":
            console.print(
                f"Current mode: [bold green]{self.config.mode}[/bold green]\n"
                f"[bold green]/y[/bold green] to switch to [bold yellow]yolo[/bold yellow] mode (execute LM commands without confirmation)\n"
                f"[bold green]/c[/bold green] to switch to [bold yellow]confirmation[/bold yellow] mode (ask for confirmation before executing LM commands)\n"
                f"[bold green]/u[/bold green] to switch to [bold yellow]human[/bold yellow] mode (execute commands issued by the user)\n"
            )
            return self._prompt_and_handle_slash_commands(prompt)
        if user_input in self._MODE_COMMANDS_MAPPING:
            if self.config.mode == self._MODE_COMMANDS_MAPPING[user_input]:
                return self._prompt_and_handle_slash_commands(
                    f"[bold red]Already in {self.config.mode} mode.[/bold red]\n{prompt}"
                )
            self.config.mode = self._MODE_COMMANDS_MAPPING[user_input]
            console.print(f"Switched to [bold green]{self.config.mode}[/bold green] mode.")
            return user_input
        return user_input

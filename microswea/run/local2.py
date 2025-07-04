"""Textual TUI interface for the agent."""

import os
import re
import threading
from pathlib import Path

from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Container, Vertical, VerticalScroll
from textual.events import Key
from textual.widgets import Footer, Header, Input, Static

from microswea.agents.default import DefaultAgent, NonTerminatingException
from microswea.agents.interactive import InteractiveAgentConfig
from microswea.environments.local import LocalEnvironment
from microswea.models import get_model


def messages_to_steps(messages: list[dict]) -> list[list[dict]]:
    print("Messages to steps called and I have", len(messages), "messages: ", messages)
    steps = []
    current_step = []
    for message in messages:
        current_step.append(message)
        if message["role"] == "user":
            steps.append(current_step)
            current_step = []
    if current_step:
        # Add the last step if it's not empty
        steps.append(current_step)
    print("Messages to steps returning", len(steps), "steps")
    print("Last step:", steps[-1])
    return steps


class TextualAgent(DefaultAgent):
    def __init__(self, app: "AgentApp", *args, **kwargs):
        self.app = app
        self._initializing = True
        super().__init__(*args, config_class=InteractiveAgentConfig, **kwargs)
        self._initializing = False
        self._action_confirmed = threading.Event()
        self._confirmation_result = None

    def add_message(self, role: str, content: str):
        super().add_message(role, content)
        print("Add message called and I have", len(self.messages), "messages: ", self.messages)
        if not self._initializing:
            self.app.call_from_thread(self.app.update_content)

    def run(self) -> str:
        try:
            result = super().run()
        finally:
            self.app.call_from_thread(self.app.set_finished)
        return result

    def execute_action(self, action: str) -> str:
        # Show confirmation before executing
        if self.config.confirm_actions and not any(re.match(r, action) for r in self.config.whitelist_actions):
            self._action_confirmed.clear()
            self._confirmation_result = None
            self.app.call_from_thread(self.app.show_confirmation, action)
            self._action_confirmed.wait()

            if self._confirmation_result:
                raise NonTerminatingException(f"Command not executed: {self._confirmation_result}")

        return super().execute_action(action)

    def set_confirmation_result(self, result: str | None):
        self._confirmation_result = result
        self._action_confirmed.set()


class MessageContainer(Vertical):
    def __init__(self, role: str, content: str):
        super().__init__(classes="message-container")
        self.role = role
        self.content = content

    def compose(self) -> ComposeResult:
        yield Static(self.role, classes="message-header")
        yield Static(self.content, classes="message-content")


class ConfirmationPrompt(Static):
    def __init__(self):
        super().__init__("Press Enter to confirm action, or DELETE to reject", classes="confirmation-prompt")
        self.rejection_mode = False
        self.can_focus = True

    def on_mount(self) -> None:
        """Focus self when mounted."""
        self.focus()

    def on_key(self, event: Key) -> None:
        app = self.screen.app
        if isinstance(app, AgentApp) and not self.rejection_mode:
            if event.key == "enter":
                event.prevent_default()
                app.confirm_action(None)
            elif event.key == "delete":
                event.prevent_default()
                self.rejection_mode = True
                self.update("Enter rejection message (optional) and press Enter:")
                app.query_one("#rejection-input", Input).display = True
                app.query_one("#rejection-input", Input).focus()


class AgentApp(App):
    BINDINGS = [
        Binding("r", "start_agent", "Start Agent"),
        Binding("q", "quit", "Quit"),
        Binding("y", "toggle_yolo", "Toggle YOLO Mode"),
    ]

    def __init__(self, model, env, problem_statement: str):
        # Load CSS from file, with environment variable override support
        css_path = os.environ.get(
            "MSWEA_LOCAL2_STYLE_PATH", str(Path(__file__).parent.parent / "config" / "local2.tcss")
        )
        self.__class__.CSS = Path(css_path).read_text()

        super().__init__()
        self.auto_follow = True
        self.agent = TextualAgent(app=self, model=model, env=env, problem_statement=problem_statement)
        self.i_step = 0
        self._agent_running = False
        self.title = "micro-SWE-agent"
        self._confirming_action = None

    def compose(self) -> ComposeResult:
        yield Header()
        with Container(id="main"):
            with VerticalScroll():
                yield Vertical(id="content")
                with Container(id="confirmation-container"):
                    yield ConfirmationPrompt()
                    yield Input(id="rejection-input", placeholder="Enter rejection message (optional)")
        yield Footer()

    def on_mount(self) -> None:
        self.update_content()
        self.hide_confirmation()
        self.run_agent_worker()

    def hide_confirmation(self):
        """Hide the confirmation UI."""
        self.query_one("#confirmation-container").display = False
        self.query_one("#rejection-input", Input).display = False
        self.query_one("#rejection-input", Input).value = ""
        self.query_one(ConfirmationPrompt).rejection_mode = False
        self.query_one(ConfirmationPrompt).update("Press Enter to confirm action, or DELETE to reject")

    def show_confirmation(self, action: str):
        """Show confirmation input for an action."""
        self._confirming_action = action
        self.query_one("#confirmation-container").display = True
        self.query_one(ConfirmationPrompt).focus()
        self.update_content()

    def on_input_submitted(self, event: Input.Submitted) -> None:
        """Handle rejection message submission."""
        if self._confirming_action is not None and self.query_one(ConfirmationPrompt).rejection_mode:
            self.confirm_action(event.value)

    def confirm_action(self, rejection_message: str | None):
        """Handle confirmation result."""
        if self._confirming_action is not None:
            self.hide_confirmation()
            self.agent.set_confirmation_result(rejection_message)
            self._confirming_action = None
            self.update_content()

    def action_toggle_yolo(self):
        """Toggle YOLO mode (skip confirmations)."""
        self.agent.config.confirm_actions = not self.agent.config.confirm_actions
        self.notify(f"YOLO mode {'disabled' if self.agent.config.confirm_actions else 'enabled'}")

    def update_content(self) -> None:
        items = messages_to_steps(self.agent.messages)
        n_steps = len(items)
        print("Update called and I have", n_steps, "steps: ", items)
        container = self.query_one("#content", Vertical)

        if not items:
            container.mount(Static("Waiting for agent to start..."))
            self.sub_title = "Waiting..."
            return

        if self.auto_follow and n_steps > 0:
            self.i_step = n_steps - 1

        # Clear existing content
        container.remove_children()

        # Add new messages
        for message in items[self.i_step]:
            msg_container = MessageContainer(role=message["role"].upper(), content=message["content"])
            container.mount(msg_container)

        # Update status and title
        status = "RUNNING" if self._agent_running else "STOPPED"
        cost = f"${self.agent.model.cost:.2f}"
        self.sub_title = f"Step {self.i_step + 1}/{n_steps} - {status} - Cost: {cost}"

        # Ensure header class matches running state
        header = self.query_one("Header")
        if self._agent_running:
            header.add_class("running")
        else:
            header.remove_class("running")

        self.refresh()

    def run_agent_worker(self):
        self._agent_running = True
        self.update_content()  # Update UI immediately when starting
        threading.Thread(target=self.agent.run, daemon=True).start()

    def set_finished(self):
        self._agent_running = False
        self.update_content()  # Update UI immediately when finishing


if __name__ == "__main__":
    app = AgentApp(
        model=get_model("gpt-4o"),
        env=LocalEnvironment(),
        problem_statement="Write a simple Python script that prints 'Hello, world!', then test run it, then quit using echo 'MICRO_SWE_AGENT_FINAL_OUTPUT' as a standalone command without any other output",
    )
    app.run()

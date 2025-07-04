"""
@local2.py This file serves the same purpose as @local.py with @interactive.py , but provides a completely different interface, using textual.

To get started, let's write a simple TUI class, which should do
"""

import threading

from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import VerticalScroll
from textual.widgets import Static

from microswea.agents.default import DefaultAgent
from microswea.environments.local import LocalEnvironment
from microswea.models import get_model


def messages_to_steps(messages: list[dict]) -> list[list[dict]]:
    """Want to group messages into (user -> assistant) pairs."""
    steps = []
    current_step = []
    for message in messages:
        current_step.append(message)
        if message["role"] == "user":
            steps.append(current_step)
            current_step = []
    return steps


class TextualAgent(DefaultAgent):
    def __init__(self, app: "AgentApp", *args, **kwargs):
        self.app = app
        self._initializing = True
        super().__init__(*args, **kwargs)
        self._initializing = False

    def add_message(self, role: str, content: str):
        super().add_message(role, content)
        # Don't notify during initialization to avoid UI updates before app is ready
        if not self._initializing:
            # Use call_from_thread to safely update UI from background thread
            self.app.call_from_thread(self.app.on_agent_message_added)


class AgentApp(App):
    BINDINGS = [
        Binding("r", "start_agent", "Start Agent"),
    ]

    def __init__(self, model, env, problem_statement: str):
        """Create app with agent parameters."""
        super().__init__()
        # Create the agent directly
        self.auto_follow = True  # Auto-advance to latest step
        self.agent = TextualAgent(app=self, model=model, env=env, problem_statement=problem_statement)
        self.i_step = 0
        self._agent_running = False

    def compose(self) -> ComposeResult:
        with VerticalScroll():
            yield Static(id="content", markup=False)

    def on_mount(self) -> None:
        self.update_content()
        self.run_agent_worker()

    @property
    def n_steps(self) -> int:
        return len(self.get_items())

    def get_items(self) -> list[list[dict]]:
        return messages_to_steps(self.agent.messages)

    def _show_step_simple(self, step: list[dict]) -> None:
        # Simplified view - show action and observation as plain text
        content_str = ""
        for message in step:
            content_str += f"{message['role'].capitalize()}: {message['content']}\n"

        content = self.query_one("#content")
        content.update(content_str)  # type: ignore

        status = "RUNNING" if self._agent_running else "STOPPED"
        self.app.sub_title = f"Step {self.i_step + 1}/{self.n_steps} - {status}"

    def update_content(self) -> None:
        items = self.get_items()
        if not items:
            # No steps yet, show waiting message
            content = self.query_one("#content")
            content.update("Waiting for agent to start...")  # type: ignore
            self.app.sub_title = "Waiting..."
            return None

        step = items[self.i_step]
        return self._show_step_simple(step)

    def auto_advance_to_latest(self) -> None:
        """Automatically advance to the latest step"""
        if self.auto_follow and self.n_steps > 0:
            self.i_step = self.n_steps - 1
            self.update_content()
            # Force a refresh to ensure the UI updates
            self.refresh()

    def on_agent_message_added(self):
        """Called when the agent adds a message - update UI"""
        self.auto_advance_to_latest()

    def run_agent_worker(self):
        """Run the agent to completion in a separate thread"""
        self._agent_running = True

        def agent_thread():
            self.agent.run()

        thread = threading.Thread(target=agent_thread, daemon=True)
        thread.start()


if __name__ == "__main__":
    app = AgentApp(
        model=get_model("gpt-4o"),
        env=LocalEnvironment(),
        problem_statement="Write a simple Python script that prints 'Hello, world!', then test run it, then quit as a separate action",
    )
    app.run()

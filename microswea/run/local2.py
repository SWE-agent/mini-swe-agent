"""Textual TUI interface for the agent."""

import os
import threading
from pathlib import Path

from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Container, Vertical, VerticalScroll
from textual.widgets import Footer, Header, Static

from microswea.agents.default import DefaultAgent
from microswea.environments.local import LocalEnvironment
from microswea.models import get_model


def messages_to_steps(messages: list[dict]) -> list[list[dict]]:
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
        if not self._initializing:
            self.app.call_from_thread(self.app.update_content)

    def run(self) -> str:
        try:
            result = super().run()
        finally:
            self.app.call_from_thread(self.app.set_finished)
        return result


class MessageContainer(Vertical):
    def __init__(self, role: str, content: str):
        super().__init__(classes="message-container")
        self.role = role
        self.content = content

    def compose(self) -> ComposeResult:
        yield Static(self.role, classes="message-header")
        yield Static(self.content, classes="message-content")


class AgentApp(App):
    BINDINGS = [
        Binding("r", "start_agent", "Start Agent"),
        Binding("q", "quit", "Quit"),
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

    def compose(self) -> ComposeResult:
        yield Header()
        with Container(id="main"):
            with VerticalScroll():
                yield Vertical(id="content")
        yield Footer()

    def on_mount(self) -> None:
        self.update_content()
        self.run_agent_worker()

    def update_content(self) -> None:
        items = messages_to_steps(self.agent.messages)
        container = self.query_one("#content", Vertical)

        if not items:
            container.mount(Static("Waiting for agent to start..."))
            self.sub_title = "Waiting..."
            return

        n_steps = len(items)
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
        self.sub_title = f"Step {self.i_step + 1}/{n_steps} - {status}"

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
        thread = threading.Thread(target=self.agent.run, daemon=True)
        thread.start()

    def set_finished(self):
        self._agent_running = False
        self.update_content()  # Update UI immediately when finishing


if __name__ == "__main__":
    app = AgentApp(
        model=get_model("gpt-4o"),
        env=LocalEnvironment(),
        problem_statement="Write a simple Python script that prints 'Hello, world!', then test run it, then quit using echo 'MICRO_SWE_AGENT_FINAL_OUTPUT' as a standalone command",
    )
    app.run()

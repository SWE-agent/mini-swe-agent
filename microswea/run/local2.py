"""Textual TUI interface for the agent."""

import threading

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


class MessageContainer(Vertical):
    def __init__(self, role: str, content: str):
        super().__init__(classes="message-container")
        self.role = role
        self.content = content

    def compose(self) -> ComposeResult:
        yield Static(self.role, classes="message-header")
        yield Static(self.content, classes="message-content")


class AgentApp(App):
    CSS = """
    Screen {
        layout: grid;
        grid-size: 1;
        grid-rows: 1fr 8 1fr;
    }

    #main {
        height: 100%;
        border: solid green;
        padding: 1;
    }

    Footer {
        dock: bottom;
        content-align: center middle;
    }

    .message-container {
        margin: 1;
        padding: 1;
        border: solid $primary-lighten-2;
        background: $surface;
    }

    .message-header {
        text-align: center;
        background: $primary-lighten-2;
        color: $text;
        padding: 0 1;
        text-style: bold;
    }

    .message-content {
        margin-top: 1;
        padding: 0 1;
    }
    """

    BINDINGS = [
        Binding("r", "start_agent", "Start Agent"),
        Binding("q", "quit", "Quit"),
    ]

    def __init__(self, model, env, problem_statement: str):
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

        status = "RUNNING" if self._agent_running else "STOPPED"
        self.sub_title = f"Step {self.i_step + 1}/{n_steps} - {status}"
        self.refresh()

    def run_agent_worker(self):
        self._agent_running = True
        thread = threading.Thread(target=self.agent.run, daemon=True)
        thread.start()


if __name__ == "__main__":
    app = AgentApp(
        model=get_model("gpt-4o"),
        env=LocalEnvironment(),
        problem_statement="Write a simple Python script that prints 'Hello, world!', then test run it, then quit as a separate action",
    )
    app.run()

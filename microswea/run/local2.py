"""Textual TUI interface for the agent."""

import threading

from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import VerticalScroll
from textual.widgets import Static

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


class AgentApp(App):
    BINDINGS = [Binding("r", "start_agent", "Start Agent")]

    def __init__(self, model, env, problem_statement: str):
        super().__init__()
        self.auto_follow = True
        self.agent = TextualAgent(app=self, model=model, env=env, problem_statement=problem_statement)
        self.i_step = 0
        self._agent_running = False

    def compose(self) -> ComposeResult:
        with VerticalScroll():
            yield Static(id="content", markup=False)

    def on_mount(self) -> None:
        self.update_content()
        self.run_agent_worker()

    def update_content(self) -> None:
        items = messages_to_steps(self.agent.messages)
        content = self.query_one("#content", Static)

        if not items:
            content.update("Waiting for agent to start...")
            self.sub_title = "Waiting..."
            return

        n_steps = len(items)
        if self.auto_follow and n_steps > 0:
            self.i_step = n_steps - 1

        content_str = ""
        for message in items[self.i_step]:
            content_str += f"{message['role'].capitalize()}: {message['content']}\n"

        content.update(content_str)
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

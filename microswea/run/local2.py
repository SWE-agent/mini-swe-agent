"""
@local2.py This file serves the same purpose as @local.py with @interactive.py , but provides a completely different interface, using textual.

To get started, let's write a simple TUI class, which should do
"""

import asyncio

from textual import work
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import VerticalScroll
from textual.widgets import Static

from microswea.agents.default import DefaultAgent, NonTerminatingException, TerminatingException
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


class AgentApp(App):
    BINDINGS = [
        Binding("right,l", "next_step", "Step++"),
        Binding("left,h", "previous_step", "Step--"),
        Binding("0", "first_step", "Step=0"),
        Binding("$", "last_step", "Step=-1"),
        Binding("j,down", "scroll_down", "Scroll down"),
        Binding("k,up", "scroll_up", "Scroll up"),
        Binding("r", "start_agent", "Start/Restart Agent"),
    ]

    def __init__(self, model, env, problem_statement: str):
        """Create app with agent parameters."""
        super().__init__()
        # Create the agent directly
        self.agent = DefaultAgent(model=model, env=env, problem_statement=problem_statement)
        self.i_step = 0
        self.auto_follow = True  # Auto-advance to latest step
        self._agent_running = False

    def compose(self) -> ComposeResult:
        with VerticalScroll():
            yield Static(id="content", markup=False)

    def on_mount(self) -> None:
        self.update_content()
        # Start the agent automatically
        self.action_start_agent()

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

        # Ensure i_step is within bounds
        if self.i_step >= len(items):
            self.i_step = len(items) - 1
        elif self.i_step < 0:
            self.i_step = 0

        step = items[self.i_step]
        return self._show_step_simple(step)

    def auto_advance_to_latest(self) -> None:
        """Automatically advance to the latest step"""
        if self.auto_follow and self.n_steps > 0:
            self.i_step = self.n_steps - 1
            self.scroll_top()
            self.update_content()
            # Force a refresh to ensure the UI updates
            self.refresh()

    @work(exclusive=True)
    async def run_agent_worker(self):
        """Run the agent step by step"""
        self._agent_running = True

        while True:
            try:
                print("stepping")
                self.agent.step()
                print("step done")
                # Yield control to allow UI updates
                await asyncio.sleep(0.1)
            except TerminatingException as e:
                self._agent_finished(str(e))
                break
            except NonTerminatingException as e:
                self.agent.add_message("user", str(e))
                # Yield control to allow UI updates
                await asyncio.sleep(0.1)
            finally:
                # Update UI from worker thread
                print("advancing")
                self.auto_advance_to_latest()
                print("advanced")

    def _agent_finished(self, final_output: str) -> None:
        """Called when agent finishes successfully"""
        self._agent_running = False
        self.auto_advance_to_latest()
        self.app.sub_title = f"Step {self.i_step + 1}/{self.n_steps} - FINISHED"

    def action_start_agent(self) -> None:
        """Start or restart the agent"""
        if not self._agent_running:
            self.run_agent_worker()

    # -----------------------------
    # UI actions
    # -----------------------------

    def action_next_step(self) -> None:
        if self.i_step < self.n_steps - 1:
            self.i_step += 1
            self.scroll_top()
            self.update_content()

    def action_previous_step(self) -> None:
        if self.i_step > 0:
            self.i_step -= 1
            self.scroll_top()
            self.update_content()

    def action_first_step(self) -> None:
        self.i_step = 0
        self.update_content()

    def action_last_step(self) -> None:
        if self.n_steps > 0:
            self.i_step = self.n_steps - 1
            self.update_content()

    def scroll_top(self) -> None:
        """Resets scrolling viewport"""
        vs = self.query_one(VerticalScroll)
        vs.scroll_home(animate=False)

    def action_scroll_down(self) -> None:
        vs = self.query_one(VerticalScroll)
        vs.scroll_to(y=vs.scroll_target_y + 15)

    def action_scroll_up(self) -> None:
        vs = self.query_one(VerticalScroll)
        vs.scroll_to(y=vs.scroll_target_y - 15)


if __name__ == "__main__":
    app = AgentApp(
        model=get_model(),
        env=LocalEnvironment(),
        problem_statement="Write a simple Python script that prints 'Hello, world!', then test run it, then quit as a separate action",
    )
    app.run()

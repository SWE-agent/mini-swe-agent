"""
Extension of the `default.py` agent that uses Textual for an interactive TUI.
For a simpler version of an interactive UI that does not require threading and more, see `interactive.py`.
"""

import logging
import os
import re
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from rich.spinner import Spinner
from rich.text import Text
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Container, Vertical, VerticalScroll
from textual.css.query import NoMatches
from textual.events import Key
from textual.widgets import Footer, Header, Static, TextArea

from microsweagent.agents.default import AgentConfig, DefaultAgent, NonTerminatingException


@dataclass
class TextualAgentConfig(AgentConfig):
    mode: Literal["confirm", "yolo"] = "confirm"
    """Mode for action execution: 'confirm' requires user confirmation, 'yolo' executes immediately."""
    whitelist_actions: list[str] = field(default_factory=list)
    """Never confirm actions that match these regular expressions."""


class TextualAgent(DefaultAgent):
    def __init__(self, app: "AgentApp", *args, **kwargs):
        """Connects the DefaultAgent to the TextualApp."""
        self.app = app
        super().__init__(*args, config_class=TextualAgentConfig, **kwargs)

    def add_message(self, role: str, content: str):
        super().add_message(role, content)
        if self.app.agent_state != "UNINITIALIZED":
            self.app.call_from_thread(self.app.on_message_added)

    def run(self, task: str) -> tuple[str, str]:
        try:
            exit_status, result = super().run(task)
        except Exception as e:
            result = str(e)
            self.app.call_from_thread(self.app.on_agent_finished, "ERROR", result)
            return "ERROR", result
        else:
            self.app.call_from_thread(self.app.on_agent_finished, exit_status, result)
        return exit_status, result

    def execute_action(self, action: dict) -> dict:
        if self.config.mode == "confirm" and not any(
            re.match(r, action["action"]) for r in self.config.whitelist_actions
        ):
            if result := self.app.confirmation_container.request_confirmation(action["action"]):
                raise NonTerminatingException(f"Command not executed: {result}")
        return super().execute_action(action)


class AddLogEmitCallback(logging.Handler):
    def __init__(self, callback):
        """Custom log handler that forwards messages via callback."""
        super().__init__()
        self.callback = callback

    def emit(self, record: logging.LogRecord):
        self.callback(record)  # type: ignore[attr-defined]


def _messages_to_steps(messages: list[dict]) -> list[list[dict]]:
    """Group messages into "pages" as shown by the UI."""
    steps = []
    current_step = []
    for message in messages:
        current_step.append(message)
        if message["role"] == "user":
            steps.append(current_step)
            current_step = []
    if current_step:
        steps.append(current_step)
    return steps


class ConfirmationPromptContainer(Container):
    def __init__(self, app: "AgentApp"):
        """This class is responsible for handling the action execution confirmation."""
        super().__init__(id="confirmation-container")
        self._app = app
        self.rejecting = False
        self.can_focus = True
        self.display = False

        self._pending_action: str | None = None
        self._confirmation_event = threading.Event()
        self._confirmation_result: str | None = None

    def compose(self) -> ComposeResult:
        yield Static(
            "Press [bold]ENTER[/bold] to confirm action or [bold]BACKSPACE[/bold] to reject (or [bold]y[/bold] to toggle YOLO mode)",
            classes="confirmation-prompt",
        )
        yield TextArea(id="rejection-input")
        rejection_help = Static(
            "Press [bold]Ctrl+D[/bold] to submit rejection message",
            id="rejection-help",
            classes="rejection-help",
        )
        rejection_help.display = False
        yield rejection_help

    def request_confirmation(self, action: str) -> str | None:
        """Request confirmation for an action. Returns rejection message or None."""
        self._confirmation_event.clear()
        self._confirmation_result = None
        self._pending_action = action
        self._app.call_from_thread(self._app.update_content)
        self._confirmation_event.wait()
        return self._confirmation_result

    def _complete_confirmation(self, rejection_message: str | None):
        """Internal method to complete the confirmation process."""
        self._confirmation_result = rejection_message
        self._pending_action = None
        self.display = False
        self.rejecting = False
        rejection_input = self.query_one("#rejection-input", TextArea)
        rejection_input.display = False
        rejection_input.text = ""
        rejection_help = self.query_one("#rejection-help", Static)
        rejection_help.display = False
        # Reset agent state to RUNNING after confirmation is completed
        if rejection_message is None:
            self._app.agent_state = "RUNNING"
        self._confirmation_event.set()
        self._app.update_content()

    def on_key(self, event: Key) -> None:
        if self.rejecting and event.key == "ctrl+d":
            event.prevent_default()
            rejection_input = self.query_one("#rejection-input", TextArea)
            self._complete_confirmation(rejection_input.text)
            return
        if not self.rejecting:
            if event.key == "enter":
                event.prevent_default()
                self._complete_confirmation(None)
            elif event.key == "backspace":
                event.prevent_default()
                self.rejecting = True
                rejection_input = self.query_one("#rejection-input", TextArea)
                rejection_input.display = True
                rejection_input.focus()
                rejection_help = self.query_one("#rejection-help", Static)
                rejection_help.display = True


class AgentApp(App):
    BINDINGS = [
        Binding("right,l", "next_step", "Step++"),
        Binding("left,h", "previous_step", "Step--"),
        Binding("0", "first_step", "Step=0"),
        Binding("$", "last_step", "Step=-1"),
        Binding("j,down", "scroll_down", "Scroll down"),
        Binding("k,up", "scroll_up", "Scroll up"),
        Binding("q", "quit", "Quit"),
        Binding("y", "yolo", "Switch to YOLO Mode"),
        Binding("c", "confirm", "Switch to Confirm Mode"),
    ]

    def __init__(self, model, env, task: str, **kwargs):
        css_path = os.environ.get(
            "MSWEA_LOCAL2_STYLE_PATH", str(Path(__file__).parent.parent / "config" / "local2.tcss")
        )
        self.__class__.CSS = Path(css_path).read_text()
        super().__init__()
        self.agent_state = "UNINITIALIZED"
        self.agent_task = task
        self.agent = TextualAgent(self, model=model, env=env, **kwargs)
        self._i_step = 0
        self.n_steps = 1
        self.confirmation_container = ConfirmationPromptContainer(self)
        self.log_handler = AddLogEmitCallback(lambda record: self.call_from_thread(self.on_log_message_emitted, record))
        logging.getLogger().addHandler(self.log_handler)
        self._spinner = Spinner("dots")
        self.exit_status: str | None = None
        self.result: str | None = None

    # --- Basics ---

    @property
    def i_step(self) -> int:
        """Current step index."""
        return self._i_step

    @i_step.setter
    def i_step(self, value: int) -> None:
        """Set current step index, automatically clamping to valid bounds."""
        if value != self._i_step:
            self._i_step = max(0, min(value, self.n_steps - 1))
            self.query_one(VerticalScroll).scroll_to(y=0, animate=False)
            self.update_content()

    def compose(self) -> ComposeResult:
        yield Header()
        with Container(id="main"):
            with VerticalScroll():
                yield Vertical(id="content")
            yield self.confirmation_container
        yield Footer()

    def on_mount(self) -> None:
        self.agent_state = "RUNNING"
        self.update_content()
        self.set_interval(1 / 8, self._update_headers)
        threading.Thread(target=lambda: self.agent.run(self.agent_task), daemon=True).start()

    # --- Reacting to events ---

    def on_message_added(self) -> None:
        vs = self.query_one(VerticalScroll)
        auto_follow = self.i_step == self.n_steps - 1 and vs.scroll_target_y <= 1
        self.n_steps = len(_messages_to_steps(self.agent.messages))
        self.update_content()
        if auto_follow:
            self.action_last_step()

    def on_log_message_emitted(self, record: logging.LogRecord) -> None:
        """Handle log messages of warning level or higher by showing them as notifications."""
        if record.levelno >= logging.WARNING:
            self.notify(f"[{record.levelname}] {record.getMessage()}", severity="warning")

    def on_unmount(self) -> None:
        """Clean up the log handler when the app shuts down."""
        if hasattr(self, "log_handler"):
            logging.getLogger().removeHandler(self.log_handler)

    def on_agent_finished(self, exit_status: str, result: str):
        self.agent_state = "STOPPED"
        self.notify(f"Agent finished with status: {exit_status}")
        self.exit_status = exit_status
        self.result = result
        self.update_content()

    # --- UI update logic ---

    def update_content(self) -> None:
        container = self.query_one("#content", Vertical)
        container.remove_children()
        items = _messages_to_steps(self.agent.messages)

        if not items:
            container.mount(Static("Waiting for agent to start..."))
            return

        for message in items[self.i_step]:
            if isinstance(message["content"], list):
                content_str = "\n".join([item["text"] for item in message["content"]])
            else:
                content_str = str(message["content"])
            message_container = Vertical(classes="message-container")
            container.mount(message_container)
            role = message["role"].replace("assistant", "micro-swe-agent")
            message_container.mount(Static(role.upper(), classes="message-header"))
            message_container.mount(Static(Text(content_str, no_wrap=False), classes="message-content"))

        if self.confirmation_container._pending_action is not None:
            self.agent_state = "AWAITING_CONFIRMATION"
        self.confirmation_container.display = (
            self.confirmation_container._pending_action is not None and self.i_step == len(items) - 1
        )
        if self.confirmation_container.display:
            self.confirmation_container.focus()

        self._update_headers()
        self.refresh()

    def _update_headers(self) -> None:
        """Update just the title with current state and spinner if needed."""
        status_text = self.agent_state
        if self.agent_state == "RUNNING":
            spinner_frame = str(self._spinner.render(time.time())).strip()
            status_text = f"{self.agent_state} {spinner_frame}"
        self.title = f"Step {self.i_step + 1}/{self.n_steps} - {status_text} - Cost: ${self.agent.model.cost:.2f}"
        try:
            self.query_one("Header").set_class(self.agent_state == "RUNNING", "running")
        except NoMatches:  # might be called when shutting down
            pass

    # --- Textual bindings ---

    def action_yolo(self):
        self.agent.config.mode = "yolo"
        self.confirmation_container._complete_confirmation(None)
        self.notify("YOLO mode enabled - actions will execute immediately")

    def action_confirm(self):
        self.agent.config.mode = "confirm"
        self.notify("Confirm mode enabled - actions will require confirmation")

    def action_next_step(self) -> None:
        self.i_step += 1

    def action_previous_step(self) -> None:
        self.i_step -= 1

    def action_first_step(self) -> None:
        self.i_step = 0

    def action_last_step(self) -> None:
        self.i_step = self.n_steps - 1

    def action_scroll_down(self) -> None:
        vs = self.query_one(VerticalScroll)
        vs.scroll_to(y=vs.scroll_target_y + 15)

    def action_scroll_up(self) -> None:
        vs = self.query_one(VerticalScroll)
        vs.scroll_to(y=vs.scroll_target_y - 15)

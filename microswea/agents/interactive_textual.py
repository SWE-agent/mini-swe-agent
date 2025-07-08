#!/usr/bin/env python3
import logging
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


class TextualAgent(DefaultAgent):
    def __init__(self, app: "AgentApp", *args, **kwargs):
        self.app = app
        self._initializing = True
        super().__init__(*args, config_class=InteractiveAgentConfig, **kwargs)
        self._initializing = False

    def add_message(self, role: str, content: str):
        super().add_message(role, content)
        if not self._initializing and self.app._app_running:
            self.app.call_from_thread(self.app.on_message_added)

    def run(self) -> str:
        try:
            result = super().run()
        finally:
            if self.app._app_running:
                self.app.call_from_thread(self.app.set_finished)
        return result

    def execute_action(self, action: str) -> str:
        if self.config.confirm_actions and not any(re.match(r, action) for r in self.config.whitelist_actions):
            result = self.app.confirmation_container.request_confirmation(action)
            if result:
                raise NonTerminatingException(f"Command not executed: {result}")

        return super().execute_action(action)


class AddLogEmitCallback(logging.Handler):
    def __init__(self, callback):
        """Custom log handler that forwards messages via callback."""
        super().__init__()
        self.callback = callback

    def emit(self, record: logging.LogRecord):
        self.callback(record)


def _messages_to_steps(messages: list[dict]) -> list[list[dict]]:
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


class MessageContainer(Vertical):
    def __init__(self, role: str, content: str):
        super().__init__(classes="message-container")
        self.role = role
        self.content = content

    def compose(self) -> ComposeResult:
        yield Static(self.role, classes="message-header")
        yield Static(self.content, classes="message-content")


class ConfirmationPromptContainer(Container):
    def __init__(self, app: "AgentApp"):
        """This class is responsible for handling the action execution confirmation."""
        super().__init__(id="confirmation-container")
        self._app = app
        self.rejecting = False
        self.can_focus = True

        # Confirmation state
        self._pending_action: str | None = None
        self._confirmation_event = threading.Event()
        self._confirmation_result: str | None = None

    def compose(self) -> ComposeResult:
        yield Static(
            "Press Enter to confirm action or BACKSPACE to reject or y to toggle YOLO mode",
            classes="confirmation-prompt",
        )
        yield Input(id="rejection-input", placeholder="Enter rejection message (optional) and/or press Enter")

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
        rejection_input = self.query_one("#rejection-input", Input)
        rejection_input.display = False
        rejection_input.value = ""
        self._confirmation_event.set()
        self._app.update_content()

    def on_key(self, event: Key) -> None:
        if not self.rejecting:
            if event.key == "enter":
                event.prevent_default()
                self._complete_confirmation(None)
            elif event.key == "backspace":
                event.prevent_default()
                self.rejecting = True
                rejection_input = self.query_one("#rejection-input", Input)
                rejection_input.display = True
                rejection_input.focus()

    def on_input_submitted(self, event: Input.Submitted) -> None:
        if self.rejecting:
            self._complete_confirmation(event.value)


class AgentApp(App):
    BINDINGS = [
        Binding("right,l", "next_step", "Step++"),
        Binding("left,h", "previous_step", "Step--"),
        Binding("0", "first_step", "Step=0"),
        Binding("$", "last_step", "Step=-1"),
        Binding("j,down", "scroll_down", "Scroll down"),
        Binding("k,up", "scroll_up", "Scroll up"),
        Binding("q", "quit", "Quit"),
        Binding("y", "toggle_yolo", "Toggle YOLO Mode"),
    ]

    def __init__(self, model, env, problem_statement: str, confirm_actions: bool):
        css_path = os.environ.get(
            "MSWEA_LOCAL2_STYLE_PATH", str(Path(__file__).parent.parent / "config" / "local2.tcss")
        )
        self.__class__.CSS = Path(css_path).read_text()

        super().__init__()

        self._app_running = False

        self.agent = TextualAgent(
            self, model=model, env=env, problem_statement=problem_statement, confirm_actions=confirm_actions
        )

        self._i_step = 0
        self.n_steps = 1
        self._agent_running = False
        self.title = "micro-SWE-agent"

        self.confirmation_container = ConfirmationPromptContainer(self)

        self.log_handler = AddLogEmitCallback(lambda record: self.call_from_thread(self.on_log_message_emitted, record))
        logging.getLogger().addHandler(self.log_handler)

    @property
    def i_step(self) -> int:
        """Current step index."""
        return self._i_step

    @i_step.setter
    def i_step(self, value: int) -> None:
        """Set current step index, automatically clamping to valid bounds."""
        if value != self._i_step:
            self._i_step = max(0, min(value, self.n_steps - 1))
            self.scroll_top()
            self.update_content()

    def compose(self) -> ComposeResult:
        yield Header()
        with Container(id="main"):
            with VerticalScroll():
                yield Vertical(id="content")
                yield self.confirmation_container
        yield Footer()

    def on_mount(self) -> None:
        self._app_running = True
        self.update_content()
        self.run_agent_worker()

    def scroll_top(self) -> None:
        self.query_one(VerticalScroll).scroll_to(y=0, animate=False)

    def on_message_added(self) -> None:
        auto_follow = self.i_step == self.n_steps - 1
        items = _messages_to_steps(self.agent.messages)
        n_steps = len(items)
        self.n_steps = n_steps

        self.update_content()
        if auto_follow:
            print("auto follow triggred")
            self.action_last_step()

    def update_content(self) -> None:
        container = self.query_one("#content", Vertical)
        items = _messages_to_steps(self.agent.messages)

        if not items:
            container.mount(Static("Waiting for agent to start..."))
            return

        container.remove_children()

        for message in items[self.i_step]:
            if isinstance(message["content"], list):
                content_str = "\n".join([item["text"] for item in message["content"]])
            else:
                content_str = str(message["content"])
            container.mount(MessageContainer(role=message["role"].upper(), content=content_str))

        # Show confirmation container if we have a pending action and we're on the latest step
        show_confirmation_prompt = (
            self.confirmation_container._pending_action is not None and self.i_step == len(items) - 1
        )
        if show_confirmation_prompt:
            self.confirmation_container.display = True
            self.confirmation_container.focus()

        status = "RUNNING" if self._agent_running and self.confirmation_container._pending_action is None else "STOPPED"
        cost = f"${self.agent.model.cost:.2f}"
        self.sub_title = f"Step {self.i_step + 1}/{len(items)} - {status} - Cost: {cost}"

        header = self.query_one("Header")
        if self._agent_running and self.confirmation_container._pending_action is None:
            header.add_class("running")
        else:
            header.remove_class("running")

        self.refresh()

    def run_agent_worker(self):
        self._agent_running = True
        self.update_content()
        threading.Thread(target=self.agent.run, daemon=True).start()

    def set_finished(self):
        self._agent_running = False
        self.update_content()

    def on_log_message_emitted(self, record: logging.LogRecord) -> None:
        """Handle log messages of warning level or higher by showing them as notifications."""
        self.notify(f"[{record.levelname}] {record.getMessage()}", severity="warning")

    def on_unmount(self) -> None:
        """Clean up the log handler when the app shuts down."""
        if hasattr(self, "log_handler"):
            logging.getLogger().removeHandler(self.log_handler)

    # --- Textual bindings ---

    def action_toggle_yolo(self):
        self.agent.config.confirm_actions = not self.agent.config.confirm_actions
        self.notify(f"YOLO mode {'disabled' if self.agent.config.confirm_actions else 'enabled'}")

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

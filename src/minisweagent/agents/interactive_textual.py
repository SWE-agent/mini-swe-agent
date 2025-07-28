"""
Extension of the `default.py` agent that uses Textual for an interactive TUI.
For a simpler version of an interactive UI that does not require threading and more, see `interactive.py`.
"""

import logging
import os
import re
import threading
import time
import traceback
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
from textual.widgets import Footer, Header, Input, Static, TextArea

from minisweagent.agents.default import AgentConfig, DefaultAgent, NonTerminatingException


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
            print(traceback.format_exc())
            self.app.call_from_thread(self.app.on_agent_finished, "ERROR", result)
            return "ERROR", result
        else:
            self.app.call_from_thread(self.app.on_agent_finished, exit_status, result)
        return exit_status, result

    def execute_action(self, action: dict) -> dict:
        if self.config.mode == "confirm" and not any(
            re.match(r, action["action"]) for r in self.config.whitelist_actions
        ):
            result = self.app.input_container.request_input("Press ENTER to confirm or provide rejection reason")
            if result:  # Non-empty string means rejection
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


class SmartInputContainer(Container):
    def __init__(self, app: "AgentApp"):
        """Smart input container supporting single-line and multi-line input modes."""
        super().__init__(id="input-container")
        self._app = app
        self._multiline_mode = False
        self.can_focus = True
        self.display = False

        # Threading state (renamed from confirmation specific names)
        self._pending_prompt: str | None = None
        self._input_event = threading.Event()
        self._input_result: str | None = None

        # Create UI elements
        self.prompt_display = Static("", id="prompt-display", classes="prompt-display")
        self.mode_indicator = Static(
            "Single-line mode ([bold]Enter[/bold] to submit, [bold]Escape[/bold] to switch to multi-line)",
            id="mode-indicator",
            classes="mode-indicator",
        )
        self.single_input = Input(placeholder="Type your input...", id="single-input")
        self.multi_input = TextArea("", show_line_numbers=False, id="multi-input")

        # Container for input elements
        self.input_elements_container = Container(
            self.prompt_display, self.mode_indicator, self.single_input, self.multi_input, id="input-elements-container"
        )

    def compose(self) -> ComposeResult:
        with Vertical(classes="message-container"):
            yield Static("[yellow]USER INPUT REQUESTED[/]", classes="message-header")
            yield self.input_elements_container

    def on_mount(self) -> None:
        """Initialize the widget state."""
        print("SmartInputContainer mounted")
        self.multi_input.display = False
        self._update_mode_display()

    def on_focus(self) -> None:
        """Called when the container gains focus."""
        print("SmartInputContainer gained focus")
        if self._multiline_mode:
            print("Focusing multi_input")
            self.multi_input.focus()
        else:
            print("Focusing single_input")
            self.single_input.focus()

    def request_input(self, prompt: str) -> str:
        """Request input from user. Returns input text (empty string if confirmed without reason)."""
        self._input_event.clear()
        self._input_result = None
        self._pending_prompt = prompt
        self.prompt_display.update(prompt)
        self._update_mode_display()  # Update the mode indicator separately
        self._app.call_from_thread(self._app.update_content)
        self._input_event.wait()
        return self._input_result or ""

    def _complete_input(self, input_text: str):
        """Internal method to complete the input process."""
        print(f"_complete_input called with: '{input_text}'")
        self._input_result = input_text
        self._pending_prompt = None
        self.prompt_display.update("")  # Clear the prompt
        self.display = False
        self.single_input.value = ""
        self.multi_input.text = ""
        self._multiline_mode = False
        self._update_mode_display()
        # Reset agent state to RUNNING after input is completed
        self._app.agent_state = "RUNNING"
        self._input_event.set()
        print("Input event set, should continue agent")
        self._app.update_content()

    def action_toggle_mode(self) -> None:
        """Toggle between single-line and multi-line modes."""
        # Only toggle if we have a pending prompt (input is active)
        if self._pending_prompt is None:
            return

        self._multiline_mode = not self._multiline_mode
        self._update_mode_display()
        self.on_focus()  # Focus the appropriate input after mode change

    def _update_mode_display(self) -> None:
        """Update the display based on current mode."""
        if self._multiline_mode:
            print("Enable Multiline mode")
            self.multi_input.text = self.single_input.value
            self.single_input.display = False
            self.multi_input.display = True

            self.mode_indicator.update(
                "Multi-line mode ([bold]Ctrl+D[/bold] to submit, [bold]Escape[/bold] to switch to single-line)"
            )
        else:
            print("Enable Singleline mode")
            self.single_input.value = "".join(self.multi_input.text.splitlines()[:1])
            self.multi_input.display = False
            self.single_input.display = True

            self.mode_indicator.update(
                "Single-line mode ([bold]Enter[/bold] to submit, [bold]Escape[/bold] to switch to multi-line)"
            )

    def on_input_submitted(self, event: Input.Submitted) -> None:
        """Handle single-line input submission."""
        if not self._multiline_mode and event.input.id == "single-input":
            text = event.input.value.strip()
            # Empty submission means confirmation, non-empty means rejection reason
            self._complete_input(text)

    def on_key(self, event: Key) -> None:
        """Handle key events."""
        if event.key == "escape":
            print("Mode toggle key pressed")
            event.prevent_default()
            event.stop()
            self.action_toggle_mode()
            return

        if self._multiline_mode and event.key == "ctrl+d":
            print("Multiline mode submit")
            event.prevent_default()
            event.stop()
            text = self.multi_input.text.strip()
            print(f"Text captured: '{text}'")
            # Empty submission means confirmation, non-empty means rejection reason
            self._complete_input(text)


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
        css_path = os.environ.get("MSWEA_MINI_STYLE_PATH", str(Path(__file__).parent.parent / "config" / "mini.tcss"))
        self.__class__.CSS = Path(css_path).read_text()
        super().__init__()
        self.agent_state = "UNINITIALIZED"
        self.agent_task = task
        self.agent = TextualAgent(self, model=model, env=env, **kwargs)
        self._i_step = 0
        self.n_steps = 1
        self.input_container = SmartInputContainer(self)
        self.log_handler = AddLogEmitCallback(lambda record: self.call_from_thread(self.on_log_message_emitted, record))
        logging.getLogger().addHandler(self.log_handler)
        self._spinner = Spinner("dots")
        self.exit_status: str | None = None
        self.result: str | None = None

        self._vscroll = VerticalScroll()

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
            self._vscroll.scroll_to(y=0, animate=False)
            self.update_content()

    def compose(self) -> ComposeResult:
        yield Header()
        with Container(id="main"):
            with self._vscroll:
                with Vertical(id="content"):
                    pass
                yield self.input_container
        yield Footer()

    def on_mount(self) -> None:
        self.agent_state = "RUNNING"
        self.update_content()
        self.set_interval(1 / 8, self._update_headers)
        threading.Thread(target=lambda: self.agent.run(self.agent_task), daemon=True).start()

    # --- Reacting to events ---

    def on_message_added(self) -> None:
        auto_follow = self.i_step == self.n_steps - 1 and self._vscroll.scroll_target_y <= 1
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
            role = message["role"].replace("assistant", "mini-swe-agent")
            message_container.mount(Static(role.upper(), classes="message-header"))
            message_container.mount(Static(Text(content_str, no_wrap=False), classes="message-content"))

        if self.input_container._pending_prompt is not None:
            self.agent_state = "AWAITING_INPUT"
        self.input_container.display = (
            self.input_container._pending_prompt is not None and self.i_step == len(items) - 1
        )
        if self.input_container.display:
            self.input_container.on_focus()

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
        self.input_container._complete_input("")
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
        self._vscroll.scroll_to(y=self._vscroll.scroll_target_y + 15)

    def action_scroll_up(self) -> None:
        self._vscroll.scroll_to(y=self._vscroll.scroll_target_y - 15)

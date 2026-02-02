#!/usr/bin/env python3
"""
Simple trajectory inspector for browsing agent conversation trajectories.

More information about the usage: [bold green] https://mini-swe-agent.com/latest/usage/inspector/ [/bold green].
"""

import json
import os
import subprocess
import tempfile
from pathlib import Path

import typer
from rich.text import Text
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.command import DiscoveryHit, Hit, Hits, Provider
from textual.containers import Container, Vertical, VerticalScroll
from textual.widgets import Footer, Header, Static

from minisweagent.models.utils.content_string import get_content_string


def _messages_to_steps(messages: list[dict]) -> list[list[dict]]:
    """Group messages into "pages" as shown by the UI."""
    steps = []
    current_step = []
    for message in messages:
        # Start new step with new tool uses
        if message.get("extra", {}).get("actions") or message.get("role") == "assistant":
            steps.append(current_step)
            current_step = [message]
        else:
            current_step.append(message)
    if current_step:
        steps.append(current_step)
    return steps


app = typer.Typer(rich_markup_mode="rich", add_completion=False)


class BindingCommandProvider(Provider):
    """Provide bindings as commands in the palette."""

    COMMAND_DESCRIPTIONS = {
        "next_step": "Next step in the current trajectory",
        "previous_step": "Previous step in the current trajectory",
        "first_step": "First step in the current trajectory",
        "last_step": "Last step in the current trajectory",
        "scroll_down": "Scroll down",
        "scroll_up": "Scroll up",
        "next_trajectory": "Next trajectory",
        "previous_trajectory": "Previous trajectory",
        "open_in_jless": "Open the current step in jless",
        "open_in_jless_all": "Open the entire trajectory in jless",
        "quit": "Quit the inspector",
    }

    async def discover(self) -> Hits:
        app = self.app
        for binding in app.BINDINGS:
            desc = self.COMMAND_DESCRIPTIONS.get(binding.action, binding.description)
            yield DiscoveryHit(desc, lambda b=binding: app.run_action(b.action))

    async def search(self, query: str) -> Hits:
        matcher = self.matcher(query)
        app = self.app
        for binding in app.BINDINGS:
            desc = self.COMMAND_DESCRIPTIONS.get(binding.action, binding.description)
            score = matcher.match(desc)
            if score > 0:
                yield Hit(score, matcher.highlight(desc), lambda b=binding: app.run_action(b.action))


class TrajectoryInspector(App):
    COMMANDS = {BindingCommandProvider}
    BINDINGS = [
        Binding("right,l", "next_step", "Step++"),
        Binding("left,h", "previous_step", "Step--"),
        Binding("0", "first_step", "Step=0"),
        Binding("$", "last_step", "Step=-1"),
        Binding("j,down", "scroll_down", "↓"),
        Binding("k,up", "scroll_up", "↑"),
        Binding("L", "next_trajectory", "Traj++"),
        Binding("H", "previous_trajectory", "Traj--"),
        Binding("e", "open_in_jless", "Jless"),
        Binding("E", "open_in_jless_all", "Jless (all)"),
        Binding("q", "quit", "Quit"),
    ]

    def __init__(self, trajectory_files: list[Path]):
        css_path = os.environ.get(
            "MSWEA_INSPECTOR_STYLE_PATH", str(Path(__file__).parent.parent.parent / "config" / "inspector.tcss")
        )
        self.__class__.CSS = Path(css_path).read_text()

        super().__init__()
        self.trajectory_files = trajectory_files
        self._i_trajectory = 0
        self._i_step = 0
        self.messages = []
        self.steps = []

        if trajectory_files:
            self._load_current_trajectory()

    # --- Basics ---

    @property
    def i_step(self) -> int:
        """Current step index."""
        return self._i_step

    @i_step.setter
    def i_step(self, value: int) -> None:
        """Set current step index, automatically clamping to valid bounds."""
        if value != self._i_step and self.n_steps > 0:
            self._i_step = max(0, min(value, self.n_steps - 1))
            self.query_one(VerticalScroll).scroll_to(y=0, animate=False)
            self.update_content()

    @property
    def n_steps(self) -> int:
        """Number of steps in current trajectory."""
        return len(self.steps)

    @property
    def i_trajectory(self) -> int:
        """Current trajectory index."""
        return self._i_trajectory

    @i_trajectory.setter
    def i_trajectory(self, value: int) -> None:
        """Set current trajectory index, automatically clamping to valid bounds."""
        if value != self._i_trajectory and self.n_trajectories > 0:
            self._i_trajectory = max(0, min(value, self.n_trajectories - 1))
            self._load_current_trajectory()
            self.query_one(VerticalScroll).scroll_to(y=0, animate=False)
            self.update_content()

    @property
    def n_trajectories(self) -> int:
        """Number of trajectory files."""
        return len(self.trajectory_files)

    def _load_current_trajectory(self) -> None:
        """Load the currently selected trajectory file."""
        if not self.trajectory_files:
            self.messages = []
            self.steps = []
            return

        trajectory_file = self.trajectory_files[self.i_trajectory]
        try:
            data = json.loads(trajectory_file.read_text())

            if isinstance(data, list):
                self.messages = data
            elif isinstance(data, dict) and "messages" in data:
                self.messages = data["messages"]
            else:
                raise ValueError("Unrecognized trajectory format")

            self.steps = _messages_to_steps(self.messages)
            self._i_step = 0
        except (json.JSONDecodeError, FileNotFoundError, ValueError) as e:
            self.messages = []
            self.steps = []
            self.notify(f"Error loading {trajectory_file.name}: {e}", severity="error")

    @property
    def current_trajectory_name(self) -> str:
        """Get the name of the current trajectory file."""
        if not self.trajectory_files:
            return "No trajectories"
        return self.trajectory_files[self.i_trajectory].name

    def compose(self) -> ComposeResult:
        yield Header()
        with Container(id="main"):
            with VerticalScroll():
                yield Vertical(id="content")
        yield Footer()

    def on_mount(self) -> None:
        self.update_content()

    def update_content(self) -> None:
        """Update the displayed content."""
        container = self.query_one("#content", Vertical)
        container.remove_children()

        if not self.steps:
            container.mount(Static("No trajectory loaded or empty trajectory"))
            self.title = "Trajectory Inspector - No Data"
            return

        for message in self.steps[self.i_step]:
            content_str = get_content_string(message)
            message_container = Vertical(classes="message-container")
            container.mount(message_container)
            role = message.get("role") or message.get("type") or "unknown"
            message_container.mount(Static(role.upper(), classes="message-header"))
            message_container.mount(Static(Text(content_str, no_wrap=False), classes="message-content"))

        self.title = (
            f"Trajectory {self.i_trajectory + 1}/{self.n_trajectories} - "
            f"{self.current_trajectory_name} - "
            f"Step {self.i_step + 1}/{self.n_steps}"
        )

    # --- Navigation actions ---

    def action_next_step(self) -> None:
        self.i_step += 1

    def action_previous_step(self) -> None:
        self.i_step -= 1

    def action_first_step(self) -> None:
        self.i_step = 0

    def action_last_step(self) -> None:
        self.i_step = self.n_steps - 1

    def action_next_trajectory(self) -> None:
        self.i_trajectory += 1

    def action_previous_trajectory(self) -> None:
        self.i_trajectory -= 1

    def action_scroll_down(self) -> None:
        vs = self.query_one(VerticalScroll)
        vs.scroll_to(y=vs.scroll_target_y + 15)

    def action_scroll_up(self) -> None:
        vs = self.query_one(VerticalScroll)
        vs.scroll_to(y=vs.scroll_target_y - 15)

    def _open_in_jless(self, path: Path) -> None:
        """Open file in jless."""
        with self.suspend():
            try:
                subprocess.run(["jless", path])
            except FileNotFoundError:
                self.notify("jless not found. Install with: `brew install jless`", severity="error")

    def action_open_in_jless(self) -> None:
        """Open the current step's messages in jless."""
        if not self.steps:
            self.notify("No messages to display", severity="warning")
            return
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(self.steps[self.i_step], f, indent=2)
            temp_path = Path(f.name)
        self._open_in_jless(temp_path)
        temp_path.unlink()

    def action_open_in_jless_all(self) -> None:
        """Open the entire trajectory in jless."""
        if not self.trajectory_files:
            self.notify("No trajectory to display", severity="warning")
            return
        self._open_in_jless(self.trajectory_files[self.i_trajectory])


@app.command(help=__doc__)
def main(
    path: str = typer.Argument(".", help="Directory to search for trajectory files or specific trajectory file"),
) -> None:
    path_obj = Path(path)

    if path_obj.is_file():
        trajectory_files = [path_obj]
    elif path_obj.is_dir():
        trajectory_files = sorted(path_obj.rglob("*.traj.json"))
        if not trajectory_files:
            raise typer.BadParameter(f"No trajectory files found in '{path}'")
    else:
        raise typer.BadParameter(f"Error: Path '{path}' does not exist")

    inspector = TrajectoryInspector(trajectory_files)
    inspector.run()


if __name__ == "__main__":
    app()

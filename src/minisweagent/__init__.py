"""
This file provides:

- Path settings for global config file & relative directories
- Version numbering
- Protocols for the core components of mini-swe-agent.
  By the magic of protocols & duck typing, you can pretty much ignore them,
  unless you want the static type checking.
"""

__version__ = "2.3.0"

import os
from enum import Enum
from pathlib import Path
from typing import Any, Protocol

import dotenv
from platformdirs import user_config_dir
from rich.console import Console

from minisweagent.utils.log import logger

package_dir = Path(__file__).resolve().parent


global_config_dir = Path(os.getenv("MSWEA_GLOBAL_CONFIG_DIR") or user_config_dir("mini-swe-agent"))
global_config_dir.mkdir(parents=True, exist_ok=True)
global_config_file = Path(global_config_dir) / ".env"

if not os.getenv("MSWEA_SILENT_STARTUP"):
    Console().print(
        f"👋 This is [bold green]mini-swe-agent[/bold green] version [bold green]{__version__}[/bold green].\n"
        f"Check the [bold red]v2 migration guide[/] at [bold red]https://klieret.short.gy/mini-v2-migration[/]\n"
        f"Loading global config from [bold green]'{global_config_file}'[/bold green]",
    )
dotenv.load_dotenv(dotenv_path=global_config_file)


# === Protocols ===
# You can ignore them unless you want static type checking.


class Model(Protocol):
    """Protocol for language models."""

    config: Any

    def query(self, messages: list[dict[str, str]], **kwargs) -> dict: ...

    def format_message(self, **kwargs) -> dict: ...

    def format_observation_messages(
        self, message: dict, outputs: list[dict], template_vars: dict | None = None
    ) -> list[dict]: ...

    def get_template_vars(self, **kwargs) -> dict[str, Any]: ...

    def serialize(self) -> dict: ...


class Environment(Protocol):
    """Protocol for execution environments."""

    config: Any

    def execute(self, action: dict, cwd: str = "") -> dict[str, Any]: ...

    def get_template_vars(self, **kwargs) -> dict[str, Any]: ...

    def serialize(self) -> dict: ...


class AgentObserverEvent(str, Enum):
    """Agent observer callback names used by agent implementations."""

    RUN_START = "on_run_start"
    RUN_END = "on_run_end"
    STEP_START = "on_step_start"
    STEP_END = "on_step_end"
    MODEL_START = "on_model_start"
    MODEL_END = "on_model_end"
    ACTION_START = "on_action_start"
    ACTION_END = "on_action_end"
    INTERRUPT = "on_interrupt"
    SUBMIT = "on_submit"
    ERROR = "on_error"


class AgentObserver(Protocol):
    """Protocol for optional agent lifecycle observers.

    Observers may implement any subset of these methods. Agent execution treats
    observer callbacks as best-effort telemetry and continues if a callback
    raises.
    """

    def on_run_start(self, **kwargs) -> None: ...

    def on_run_end(self, **kwargs) -> None: ...

    def on_step_start(self, **kwargs) -> None: ...

    def on_step_end(self, **kwargs) -> None: ...

    def on_model_start(self, **kwargs) -> None: ...

    def on_model_end(self, **kwargs) -> None: ...

    def on_action_start(self, **kwargs) -> None: ...

    def on_action_end(self, **kwargs) -> None: ...

    def on_interrupt(self, **kwargs) -> None: ...

    def on_submit(self, **kwargs) -> None: ...

    def on_error(self, **kwargs) -> None: ...


class Agent(Protocol):
    """Protocol for agents."""

    config: Any

    def run(self, task: str, **kwargs) -> dict: ...

    def save(self, path: Path | None, *extra_dicts) -> dict: ...


__all__ = [
    "Agent",
    "AgentObserverEvent",
    "AgentObserver",
    "Model",
    "Environment",
    "package_dir",
    "__version__",
    "global_config_file",
    "global_config_dir",
    "logger",
]

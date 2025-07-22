"""
This file provides:

- Path settings for global config file & relative directories
- Version numbering
- Protocols for the core components of micro-swe-agent.
  By the magic of protocols & duck typing, you can pretty much ignore them,
  unless you want the static type checking.
"""

__version__ = "1.0.1"

import os
from pathlib import Path
from typing import Any, Protocol

import dotenv
from platformdirs import user_config_dir
from rich.console import Console

package_dir = Path(__file__).resolve().parent

global_config_dir = Path(os.getenv("MICRO_SWE_AGENT_GLOBAL_CONFIG_DIR") or user_config_dir("micro-swe-agent"))
global_config_dir.mkdir(parents=True, exist_ok=True)
global_config_file = Path(global_config_dir) / ".env"

if not os.getenv("MICRO_SWE_AGENT_SILENT_STARTUP"):
    Console().print(
        f"👋 This is [bold green]micro-swe-agent[/bold green] version [bold green]{__version__}[/bold green].\n"
        f"Your config is stored in [bold green]'{global_config_file}'[/bold green]"
    )
dotenv.load_dotenv(dotenv_path=global_config_file)


# === Protocols ===
# You can ignore them unless you want static type checking.


class Model(Protocol):
    """Protocol for language models."""

    config: Any
    cost: float
    n_calls: int

    def query(self, messages: list[dict[str, str]], **kwargs) -> dict: ...


class Environment(Protocol):
    """Protocol for execution environments."""

    config: Any

    def execute(self, command: str, cwd: str = "") -> dict[str, str]: ...


class Agent(Protocol):
    """Protocol for agents."""

    model: Model
    env: Environment
    messages: list[dict[str, str]]

    def run(self, task: str) -> tuple[str, str]: ...


__all__ = ["Agent", "Model", "Environment", "package_dir", "__version__", "global_config_file", "global_config_dir"]

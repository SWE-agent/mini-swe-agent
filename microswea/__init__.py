"""Nano SWE Agent - A simple AI software engineering agent."""

__version__ = "1.0.0.dev3"

from pathlib import Path
from typing import Any, Protocol

import dotenv
from platformdirs import user_config_dir
from rich.console import Console

package_dir = Path(__file__).resolve().parent

global_config_file = Path(user_config_dir("micro-swe-agent", appauthor=False, ensure_exists=True)) / ".env"

Console().print(
    f"This is micro-swe-agent version [bold green]{__version__}[/bold green]\n"
    f"Your config is stored in [bold green]'{global_config_file}'[/bold green]"
)
dotenv.load_dotenv(dotenv_path=global_config_file)


class Model(Protocol):
    """Protocol for language models."""

    config: Any
    cost: float
    n_calls: int

    def query(self, messages: list[dict[str, str]], **kwargs) -> str:
        """Query the model with a list of messages and return the response."""
        ...


class Environment(Protocol):
    """Protocol for execution environments."""

    def execute(self, command: str, cwd: str = "") -> dict[str, str]:
        """Execute a command in the environment and return the raw output."""
        ...


__all__ = ["Model", "Environment", "package_dir", "__version__", "global_config_file"]

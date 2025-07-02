"""Nano SWE Agent - A simple AI software engineering agent."""

__version__ = "1.0.0.dev3"

from pathlib import Path
from typing import Any, Protocol

import dotenv

package_dir = Path(__file__).resolve().parent


dotenv.load_dotenv(dotenv_path=package_dir.parent / ".env")


class Model(Protocol):
    """Protocol for language models."""

    config: Any
    cost: float
    n_calls: int

    def query(self, messages: list[dict[str, str]]) -> str:
        """Query the model with a list of messages and return the response."""
        ...


class Environment(Protocol):
    """Protocol for execution environments."""

    def execute(self, command: str, cwd: str = "") -> dict[str, str]:
        """Execute a command in the environment and return the raw output."""
        ...


__all__ = ["Model", "Environment", "package_dir", "__version__"]

"""This file provides convenience functions for selecting models.
You can ignore this file completely if you explicitly set your model in your run script.
"""

import copy
import os
import threading

from dotenv import set_key
from rich.console import Console

from microsweagent import Model, global_config_file

console = Console()


class GlobalModelStats:
    """Global model statistics tracker with optional limits."""

    def __init__(self):
        self._cost = 0.0
        self._n_calls = 0
        self._lock = threading.Lock()

    def add(self, cost: float) -> None:
        """Add a model call with its cost, checking limits."""
        with self._lock:
            self._cost += cost
            self._n_calls += 1
            cost_limit = float(os.getenv("MSWEA_GLOBAL_COST_LIMIT", "0"))
            call_limit = int(os.getenv("MSWEA_GLOBAL_CALL_LIMIT", "0"))
        if 0 < cost_limit < self._cost or 0 < call_limit < self._n_calls + 1:
            raise RuntimeError(f"Global cost/call limit exceeded: ${self._cost:.4f} / {self._n_calls + 1}")

    @property
    def cost(self) -> float:
        return self._cost

    @property
    def n_calls(self) -> int:
        return self._n_calls


GLOBAL_MODEL_STATS = GlobalModelStats()


def get_model(input_model_name: str | None = None, config: dict | None = None) -> Model:
    """Get an initialized model object from any kind of user input or settings."""
    resolved_model_name = get_model_name(input_model_name, config)
    if config is None:
        config = {}
    config = copy.deepcopy(config)
    config["model_name"] = resolved_model_name

    # API key resolution (from env -> config -> None)
    if "model_kwargs" not in config:
        config["model_kwargs"] = {}
    if from_env := os.getenv("MSWEA_MODEL_API_KEY"):
        config["model_kwargs"]["api_key"] = from_env
    return get_model_class(resolved_model_name)(**config)


def get_model_name(input_model_name: str | None = None, config: dict | None = None) -> str:
    """Get a model name from any kind of user input or settings."""
    if config is None:
        config = {}
    if input_model_name:
        return input_model_name
    if from_env := os.getenv("MSWEA_MODEL_NAME"):
        return from_env
    if from_config := config.get("model_name"):
        return from_config
    return prompt_for_model_name()


def get_model_class(model_name: str) -> type:
    """Select the best model class for a given model name."""
    if any(s in model_name for s in ["anthropic", "sonnet", "opus"]):
        from microsweagent.models.anthropic import AnthropicModel

        return AnthropicModel
    from microsweagent.models.litellm_model import LitellmModel

    return LitellmModel


def prompt_for_model_name() -> str:
    """Prompt the user for a model name and store it in the global config file."""
    msg = (
        "[bold yellow]Choose your language model[/bold yellow]\n"
        "Popular models:\n"
        "[bold green]claude-3-5-sonnet-20241022[/bold green]\n[bold green]gpt-4o[/bold green]\n"
        "[bold yellow]Your language model: [/bold yellow]"
    )
    choice = console.input(msg)
    set_key(global_config_file, "MSWEA_MODEL_NAME", choice)
    return choice

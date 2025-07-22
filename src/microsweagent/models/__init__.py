"""This file provides convenience functions for selecting models.
You can ignore this file completely if you explicitly set your model in your run script.
"""

import copy
import os
import threading

from microsweagent import Model


class GlobalModelStats:
    """Global model statistics tracker with optional limits."""

    def __init__(self):
        self._cost = 0.0
        self._n_calls = 0
        self._lock = threading.Lock()
        self.cost_limit = float(os.getenv("MSWEA_GLOBAL_COST_LIMIT", "0"))
        self.call_limit = int(os.getenv("MSWEA_GLOBAL_CALL_LIMIT", "0"))
        if (self.cost_limit > 0 or self.call_limit > 0) and not os.getenv("MICRO_SWE_AGENT_SILENT_STARTUP"):
            print(f"Global cost/call limit: ${self.cost_limit:.4f} / {self.call_limit}")

    def add(self, cost: float) -> None:
        """Add a model call with its cost, checking limits."""
        with self._lock:
            self._cost += cost
            self._n_calls += 1
        if 0 < self.cost_limit < self._cost or 0 < self.call_limit < self._n_calls + 1:
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
    raise ValueError("No default model set. Please run `micro-extra config setup` to set one.")


def get_model_class(model_name: str) -> type:
    """Select the best model class for a given model name."""
    if any(s in model_name.lower() for s in ["anthropic", "sonnet", "opus", "claude"]):
        from microsweagent.models.anthropic import AnthropicModel

        return AnthropicModel
    from microsweagent.models.litellm_model import LitellmModel

    return LitellmModel

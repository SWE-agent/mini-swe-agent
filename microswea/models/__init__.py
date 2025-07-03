"""Model implementations for micro-SWE-agent.
You can ignore this file if you explicitly set your model in your run script.
"""


import os

from rich.console import Console

from microswea import Model


def get_model_class(model_name: str) -> type:
    if any(s in model_name for s in ["anthropic", "sonnet", "opus"]):
        from microswea.models.anthropic import AnthropicModel

        return AnthropicModel
    from microswea.models.litellm_model import LitellmModel

    return LitellmModel


def get_model_name(input_model_name: str |None = None, config: dict | None = None) -> str:
    if config is None:
        config = {}
    if input_model_name:
        return input_model_name
    if from_env := os.getenv("MSWEA_MODEL_NAME"):
        return from_env
    if from_config := config.get("model", {}).get("model_name"):
        return from_config
    return Console().input("[bold yellow]Enter your model name: [/bold yellow]")


def get_model(model_name: str | None = None, config: dict | None = None) -> Model:
    resolved_model_name = get_model_name(model_name, config)
    if config is None:
        config = {}
    return get_model_class(resolved_model_name)(**(config.get("model", {}) | {"model_name": resolved_model_name}))
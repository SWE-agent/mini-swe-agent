import os
import warnings
from typing import Literal

from minisweagent.models.litellm_model import LitellmModel, LitellmModelConfig
from minisweagent.models.utils.cache_control import set_cache_control
from minisweagent.models.utils.key_per_thread import get_key_per_thread


class AnthropicModelConfig(LitellmModelConfig):
    set_cache_control: Literal["default_end"] | None = "default_end"
    """Set explicit cache control markers, for example for Anthropic models"""


class AnthropicModel(LitellmModel):
    """This class is now only a thin wrapper around the LitellmModel class.
    It is largely kept for backwards compatibility.
    It will not be selected by `get_model` and `get_model_class` unless explicitly specified.
    """

    def __init__(self, *, config_class: type = AnthropicModelConfig, **kwargs):
        super().__init__(config_class=config_class, **kwargs)

    def query(self, messages: list[dict], **kwargs) -> dict:
        api_key = None
        # Legacy only
        if rotating_keys := os.getenv("ANTHROPIC_API_KEYS"):
            warnings.warn(
                "ANTHROPIC_API_KEYS is deprecated and will be removed in the future. "
                "Simply use the ANTHROPIC_API_KEY environment variable instead. "
                "Key rotation is no longer required."
            )
            api_key = get_key_per_thread(rotating_keys.split("::"))
        messages = set_cache_control(messages, mode="default_end")
        return super().query(messages, api_key=api_key, **kwargs)

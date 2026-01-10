from typing import Literal

from minisweagent.models.litellm_model import LitellmModel, LitellmModelConfig
from minisweagent.models.utils.cache_control import set_cache_control


class AnthropicModelConfig(LitellmModelConfig):
    set_cache_control: Literal["default_end"] | None = "default_end"
    """Set explicit cache control markers, for example for Anthropic models"""


class AnthropicModel(LitellmModel):
    """This class is now only a thin wrapper around the LitellmModel class.
    It will not be selected by `get_model` and `get_model_class` unless explicitly specified.
    """

    def __init__(self, *, config_class: type = AnthropicModelConfig, **kwargs):
        super().__init__(config_class=config_class, **kwargs)

    def query(self, messages: list[dict], **kwargs) -> dict:
        api_key = None
        messages = set_cache_control(messages, mode="default_end")
        return super().query(messages, api_key=api_key, **kwargs)

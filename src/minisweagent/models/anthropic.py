import os
import warnings

from minisweagent.models.litellm_model import LitellmModel
from minisweagent.models.utils.cache_control import set_cache_control
from minisweagent.models.utils.key_per_thread import get_key_per_thread


class AnthropicModel(LitellmModel):
    """For the use of anthropic models, we need to add explicit cache control marks
    to the messages or we lose out on the benefits of the cache.
    """

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

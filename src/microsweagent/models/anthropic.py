import os

from microsweagent.models.litellm_model import LitellmModel
from microsweagent.models.utils.cache_control import set_cache_control
from microsweagent.models.utils.key_per_thread import get_key_per_thread


class AnthropicModel(LitellmModel):
    """For the use of anthropic models, we need to add explicit cache control marks
    to the messages or we lose out on the benefits of the cache.
    Because break points are limited per key, we also need to rotate between different keys
    if running with multiple agents in parallel threads.
    """

    def query(self, messages: list[dict], **kwargs) -> dict:
        api_key = None
        if rotating_keys := os.getenv("ANTHROPIC_API_KEYS"):
            api_key = get_key_per_thread(rotating_keys.split("::"))
        return super().query(set_cache_control(messages), api_key=api_key, **kwargs)

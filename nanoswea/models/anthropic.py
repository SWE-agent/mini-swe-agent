import os

from nanoswea.models.litellm_model import LitellmModel
from nanoswea.models.utils.cache_control import set_cache_control
from nanoswea.models.utils.key_per_thread import get_key_per_thread


class AnthropicModel(LitellmModel):
    """For the use of anthropic models, we need to add explicit cache control marks
    to the history or we lose out on the benefits of the cache.
    Because break points are limited per key, we also need to rotate between different keys
    if running with multiple agents in parallel threads.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def query(self, history: list[dict], **kwargs) -> str:
        api_key = None
        if rotating_keys := os.getenv("ANTHROPIC_API_KEYS"):
            api_key = get_key_per_thread(rotating_keys.split("::"))
        return super().query(set_cache_control(history), api_key=api_key, **kwargs)

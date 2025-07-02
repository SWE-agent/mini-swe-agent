from nanoswea.models.litellm_model import LitellmModel
from nanoswea.models.utils.cache_control import set_cache_control


class AnthropicModel(LitellmModel):
    """For the use of anthropic models, we need to add explicit cache control marks
    to the history or we lose out on the benefits of the cache.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def query(self, history: list[dict]) -> str:
        return super().query(set_cache_control(history))

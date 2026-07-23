import os
from typing import Literal

from minisweagent.models.litellm_model import LitellmModel, LitellmModelConfig


class AtlasCloudModelConfig(LitellmModelConfig):
    cost_tracking: Literal["default", "ignore_errors"] = "ignore_errors"
    """Atlas Cloud model costs are not in LiteLLM's bundled registry, so default to zero-cost tracking."""


class AtlasCloudModel(LitellmModel):
    """Atlas Cloud OpenAI-compatible model adapter."""

    def __init__(self, **kwargs):
        model_kwargs = dict(kwargs.pop("model_kwargs", {}))
        api_key = model_kwargs.pop("api_key", None) or os.getenv("ATLASCLOUD_API_KEY")
        if not api_key:
            msg = "ATLASCLOUD_API_KEY is required for Atlas Cloud models."
            raise ValueError(msg)

        api_base = model_kwargs.pop("api_base", None) or os.getenv(
            "ATLASCLOUD_API_BASE", "https://api.atlascloud.ai/v1"
        )
        kwargs["model_name"] = self._to_litellm_model_name(kwargs["model_name"])
        kwargs.setdefault("cost_tracking", "ignore_errors")
        kwargs["model_kwargs"] = {
            "api_key": api_key,
            "api_base": api_base,
            **model_kwargs,
        }
        super().__init__(config_class=AtlasCloudModelConfig, **kwargs)

    @staticmethod
    def _to_litellm_model_name(model_name: str) -> str:
        return f"openai/{model_name.removeprefix('atlascloud/')}"

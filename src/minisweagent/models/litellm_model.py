import json
import logging
import os
from collections.abc import Callable
from pathlib import Path
from typing import Any, Literal

import litellm
from litellm.types.utils import Choices, Message, ModelResponse, Usage
from pydantic import BaseModel
from tenacity import (
    before_sleep_log,
    retry,
    retry_if_not_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from minisweagent.models import GLOBAL_MODEL_STATS
from minisweagent.models.utils.cache_control import set_cache_control

logger = logging.getLogger("litellm_model")


class LitellmModelConfig(BaseModel):
    model_name: str
    model_kwargs: dict[str, Any] = {}
    litellm_model_registry: Path | str | None = os.getenv("LITELLM_MODEL_REGISTRY_PATH")
    set_cache_control: Literal["default_end"] | None = None
    """Set explicit cache control markers, for example for Anthropic models"""
    cost_tracking: Literal["default", "ignore_errors"] = os.getenv("MSWEA_COST_TRACKING", "default")
    """Cost tracking mode for this model. Can be "default" or "ignore_errors" (ignore errors/missing cost info)"""
    use_streaming: bool = os.getenv("MSWEA_USE_STREAMING", "true").lower() == "true"
    """Use streaming mode to avoid HTTP read timeouts on long generations. Default: true.
    When enabled, responses are streamed token-by-token, keeping the connection alive.
    This prevents timeout errors when vLLM takes >10 minutes to generate a response."""


class LitellmModel:
    def __init__(self, *, config_class: Callable = LitellmModelConfig, **kwargs):
        self.config = config_class(**kwargs)
        self.cost = 0.0
        self.n_calls = 0
        if self.config.litellm_model_registry and Path(self.config.litellm_model_registry).is_file():
            litellm.utils.register_model(json.loads(Path(self.config.litellm_model_registry).read_text()))

    def _reconstruct_response_from_stream(self, stream_response) -> ModelResponse:
        """Accumulate streaming chunks into a complete ModelResponse.

        This avoids HTTP read timeouts on long generations by keeping the
        connection alive as tokens stream in.
        """
        content_parts = []
        last_chunk = None

        for chunk in stream_response:
            last_chunk = chunk
            if chunk.choices and chunk.choices[0].delta.content:
                content_parts.append(chunk.choices[0].delta.content)

        content = "".join(content_parts)

        # Reconstruct a ModelResponse compatible with non-streaming code
        return ModelResponse(
            id=last_chunk.id if last_chunk else "stream-response",
            created=last_chunk.created if last_chunk else 0,
            model=last_chunk.model if last_chunk else self.config.model_name,
            choices=[Choices(
                index=0,
                finish_reason="stop",
                message=Message(role="assistant", content=content),
            )],
            # Usage stats not available in streaming; use estimates
            usage=Usage(
                prompt_tokens=0,
                completion_tokens=len(content_parts),
                total_tokens=len(content_parts),
            ),
        )

    @retry(
        reraise=True,
        stop=stop_after_attempt(int(os.getenv("MSWEA_MODEL_RETRY_STOP_AFTER_ATTEMPT", "10"))),
        wait=wait_exponential(multiplier=1, min=4, max=60),
        before_sleep=before_sleep_log(logger, logging.WARNING),
        retry=retry_if_not_exception_type(
            (
                litellm.exceptions.UnsupportedParamsError,
                litellm.exceptions.NotFoundError,
                litellm.exceptions.PermissionDeniedError,
                litellm.exceptions.ContextWindowExceededError,
                litellm.exceptions.APIError,
                litellm.exceptions.AuthenticationError,
                KeyboardInterrupt,
            )
        ),
    )
    def _query(self, messages: list[dict[str, str]], **kwargs):
        try:
            if self.config.use_streaming:
                # Use streaming to avoid HTTP read timeouts on long generations
                stream_response = litellm.completion(
                    model=self.config.model_name,
                    messages=messages,
                    stream=True,
                    **(self.config.model_kwargs | kwargs)
                )
                return self._reconstruct_response_from_stream(stream_response)
            else:
                # Non-streaming mode (original behavior)
                return litellm.completion(
                    model=self.config.model_name, messages=messages, **(self.config.model_kwargs | kwargs)
                )
        except litellm.exceptions.AuthenticationError as e:
            e.message += " You can permanently set your API key with `mini-extra config set KEY VALUE`."
            raise e

    def query(self, messages: list[dict[str, str]], **kwargs) -> dict:
        if self.config.set_cache_control:
            messages = set_cache_control(messages, mode=self.config.set_cache_control)
        response = self._query([{"role": msg["role"], "content": msg["content"]} for msg in messages], **kwargs)
        try:
            cost = litellm.cost_calculator.completion_cost(response, model=self.config.model_name)
            if cost <= 0.0:
                raise ValueError(f"Cost must be > 0.0, got {cost}")
        except Exception as e:
            cost = 0.0
            if self.config.cost_tracking != "ignore_errors":
                msg = (
                    f"Error calculating cost for model {self.config.model_name}: {e}, perhaps it's not registered? "
                    "You can ignore this issue from your config file with cost_tracking: 'ignore_errors' or "
                    "globally with export MSWEA_COST_TRACKING='ignore_errors'. "
                    "Alternatively check the 'Cost tracking' section in the documentation at "
                    "https://klieret.short.gy/mini-local-models. "
                    " Still stuck? Please open a github issue at https://github.com/SWE-agent/mini-swe-agent/issues/new/choose!"
                )
                logger.critical(msg)
                raise RuntimeError(msg) from e
        self.n_calls += 1
        self.cost += cost
        GLOBAL_MODEL_STATS.add(cost)
        return {
            "content": response.choices[0].message.content or "",  # type: ignore
            "extra": {
                "response": response.model_dump(),
            },
        }

    def get_template_vars(self) -> dict[str, Any]:
        return self.config.model_dump() | {"n_model_calls": self.n_calls, "model_cost": self.cost}

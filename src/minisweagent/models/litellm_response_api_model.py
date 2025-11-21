import logging
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import litellm
from openai.types.responses.response_output_message import ResponseOutputMessage
from tenacity import (
    before_sleep_log,
    retry,
    retry_if_not_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from minisweagent.models.litellm_model import LitellmModel, LitellmModelConfig

logger = logging.getLogger("litellm_response_api_model")


@dataclass
class LitellmResponseAPIModelConfig(LitellmModelConfig):
    pass


class LitellmResponseAPIModel(LitellmModel):
    def __init__(self, *, config_class: Callable = LitellmResponseAPIModelConfig, **kwargs):
        super().__init__(config_class=config_class, **kwargs)
        self._previous_response_id: str | None = None

    @retry(
        stop=stop_after_attempt(10),
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
            resp = litellm.responses(
                model=self.config.model_name,
                input=messages if self._previous_response_id is None else messages[-1:],
                previous_response_id=self._previous_response_id,
                **(self.config.model_kwargs | kwargs),
            )
            self._previous_response_id = getattr(resp, "id", None)
            return resp
        except litellm.exceptions.AuthenticationError as e:
            e.message += " You can permanently set your API key with `mini-extra config set KEY VALUE`."
            raise e

    def query(self, messages: list[dict[str, str]], **kwargs) -> dict:
        response = self._query(messages, **kwargs)
        text = self._coerce_responses_text(response)
        try:
            cost = litellm.cost_calculator.completion_cost(response)
        except Exception as e:
            logger.critical(
                f"Error calculating cost for model {self.config.model_name}: {e}. "
                "Please check the 'Updating the model registry' section in the documentation. "
                "http://bit.ly/4p31bi4 Still stuck? Please open a github issue for help!"
            )
            raise
        self.n_calls += 1
        self.cost += cost
        from minisweagent.models import GLOBAL_MODEL_STATS

        GLOBAL_MODEL_STATS.add(cost)
        return {
            "content": text,
        }

    def _coerce_responses_text(self, resp: Any) -> str:
        """Helper to normalize LiteLLM Responses API result to text."""
        # openai client directly returns `output_text`, but litellm doesn't support it yet.
        text = getattr(resp, "output_text", None)
        if isinstance(text, str) and text:
            return text

        # Concatenate all (to be consistent with openai client)
        output = [item.content[0].text for item in resp.output if isinstance(item, ResponseOutputMessage)]
        return "\n\n".join(output) or ""

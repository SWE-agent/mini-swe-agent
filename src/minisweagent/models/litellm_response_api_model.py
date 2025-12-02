import logging
from collections.abc import Callable
from dataclasses import dataclass

import litellm
from tenacity import (
    before_sleep_log,
    retry,
    retry_if_not_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from minisweagent.models.litellm_model import LitellmModel, LitellmModelConfig
from minisweagent.models.utils.openai_utils import coerce_responses_text

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
        print(response)
        text = coerce_responses_text(response)
        try:
            cost = litellm.cost_calculator.completion_cost(response, model=self.config.model_name)
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

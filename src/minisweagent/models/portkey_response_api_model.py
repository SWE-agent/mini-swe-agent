import logging
import os
from dataclasses import dataclass

import litellm
from tenacity import (
    before_sleep_log,
    retry,
    retry_if_not_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from minisweagent.models import GLOBAL_MODEL_STATS
from minisweagent.models.portkey_model import PortkeyModel, PortkeyModelConfig
from minisweagent.models.utils.cache_control import set_cache_control
from minisweagent.models.utils.openai_utils import coerce_responses_text

logger = logging.getLogger("portkey_response_api_model")


@dataclass
class PortkeyResponseAPIModelConfig(PortkeyModelConfig):
    pass


class PortkeyResponseAPIModel(PortkeyModel):
    def __init__(self, *, config_class: type = PortkeyResponseAPIModelConfig, **kwargs):
        super().__init__(config_class=config_class, **kwargs)
        self._previous_response_id: str | None = None

    @retry(
        stop=stop_after_attempt(int(os.getenv("MSWEA_MODEL_RETRY_STOP_AFTER_ATTEMPT", "10"))),
        wait=wait_exponential(multiplier=1, min=4, max=60),
        before_sleep=before_sleep_log(logger, logging.WARNING),
        retry=retry_if_not_exception_type((KeyboardInterrupt, TypeError, ValueError)),
    )
    def _query(self, messages: list[dict[str, str]], **kwargs):
        input_messages = messages if self._previous_response_id is None else messages[-1:]
        resp = self.client.responses.create(
            model=self.config.model_name,
            input=input_messages,
            previous_response_id=self._previous_response_id,
            **(self.config.model_kwargs | kwargs),
        )
        self._previous_response_id = getattr(resp, "id", None)
        return resp

    def query(self, messages: list[dict[str, str]], **kwargs) -> dict:
        if self.config.set_cache_control:
            messages = set_cache_control(messages, mode=self.config.set_cache_control)
        response = self._query(messages, **kwargs)
        text = coerce_responses_text(response)
        try:
            cost = litellm.cost_calculator.completion_cost(response, model=self.config.model_name)
            assert cost > 0.0, f"Cost is not positive: {cost}"
        except Exception as e:
            if self.config.cost_tracking != "ignore_errors":
                raise RuntimeError(
                    f"Error calculating cost for model {self.config.model_name}: {e}. "
                    "You can ignore this issue from your config file with cost_tracking: 'ignore_errors' or "
                    "globally with export MSWEA_COST_TRACKING='ignore_errors' to ignore this error. "
                ) from e
            cost = 0.0
        self.n_calls += 1
        self.cost += cost
        GLOBAL_MODEL_STATS.add(cost)
        return {
            "content": text,
            "extra": {
                "response": response.model_dump() if hasattr(response, "model_dump") else {},
                "cost": cost,
            },
        }

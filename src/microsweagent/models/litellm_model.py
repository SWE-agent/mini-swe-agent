import logging
from dataclasses import dataclass, field
from typing import Any

import litellm
from tenacity import before_sleep_log, retry, stop_after_attempt, wait_exponential

from microsweagent.models import GLOBAL_MODEL_STATS

logger = logging.getLogger("litellm_model")


@dataclass
class LitellmModelConfig:
    model_name: str
    model_kwargs: dict[str, Any] = field(default_factory=dict)


class LitellmModel:
    def __init__(self, **kwargs):
        self.config = LitellmModelConfig(**kwargs)
        self.cost = 0.0
        self.n_calls = 0

    @retry(
        stop=stop_after_attempt(5),
        wait=wait_exponential(multiplier=1, min=4, max=10),
        before_sleep=before_sleep_log(logger, logging.WARNING),
    )
    def _query(self, messages: list[dict[str, str]], **kwargs) -> str:
        response: litellm.types.utils.ModelResponse = litellm.completion(  # type: ignore
            model=self.config.model_name, messages=messages, **(self.config.model_kwargs | kwargs)
        )
        return response

    def query(self, messages: list[dict[str, str]], **kwargs) -> str:
        response = self._query(messages, **kwargs)
        cost = litellm.cost_calculator.completion_cost(response)
        self.n_calls += 1
        self.cost += cost
        GLOBAL_MODEL_STATS.add(cost)
        return response.choices[0].message.content  # type: ignore

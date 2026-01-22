import logging
import time

import litellm

from minisweagent.models import GLOBAL_MODEL_STATS
from minisweagent.models.portkey_model import PortkeyModel, PortkeyModelConfig
from minisweagent.models.utils.actions_text import parse_regex_actions
from minisweagent.models.utils.openai_response_api import _coerce_responses_text
from minisweagent.models.utils.retry import retry

logger = logging.getLogger("portkey_response_api_model")


class PortkeyResponseAPIModelConfig(PortkeyModelConfig):
    pass


class PortkeyResponseAPIModel(PortkeyModel):
    def __init__(self, *, config_class: type = PortkeyResponseAPIModelConfig, **kwargs):
        super().__init__(config_class=config_class, **kwargs)
        self._previous_response_id: str | None = None

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
        for attempt in retry(logger=logger, abort_exceptions=self.abort_exceptions):
            with attempt:
                response = self._query(self._prepare_messages_for_api(messages), **kwargs)
        content = _coerce_responses_text(response)
        cost_output = self._calculate_cost(response)
        GLOBAL_MODEL_STATS.add(cost_output["cost"])
        return {
            "role": "assistant",
            "content": content,
            "extra": {
                "actions": self._parse_actions(response),
                "response": response.model_dump() if hasattr(response, "model_dump") else {},
                **cost_output,
                "timestamp": time.time(),
            },
        }

    def _parse_actions(self, response) -> list[dict]:
        """Parse actions from the response API response. Uses coerce_responses_text for content extraction."""
        content = _coerce_responses_text(response)
        return parse_regex_actions(
            content, action_regex=self.config.action_regex, format_error_template=self.config.format_error_template
        )

    def _calculate_cost(self, response) -> dict[str, float]:
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
        return {"cost": cost}

import logging
import time
from collections.abc import Callable

import litellm

from minisweagent.models import GLOBAL_MODEL_STATS
from minisweagent.models.litellm_model import LitellmModel, LitellmModelConfig
from minisweagent.models.utils.actions_toolcall import (
    BASH_TOOL,
    format_toolcall_observation_messages,
    parse_toolcall_actions,
)
from minisweagent.models.utils.retry import retry

logger = logging.getLogger("litellm_response_toolcall_model")


class LitellmResponseToolcallModelConfig(LitellmModelConfig):
    format_error_template: str = "{{ error }}"


class LitellmResponseToolcallModel(LitellmModel):
    def __init__(self, *, config_class: Callable = LitellmResponseToolcallModelConfig, **kwargs):
        super().__init__(config_class=config_class, **kwargs)
        self._previous_response_id: str | None = None

    def _query(self, messages: list[dict[str, str]], **kwargs):
        try:
            resp = litellm.responses(
                model=self.config.model_name,
                input=messages if self._previous_response_id is None else messages[-1:],
                previous_response_id=self._previous_response_id,
                tools=[BASH_TOOL],
                **(self.config.model_kwargs | kwargs),
            )
            self._previous_response_id = getattr(resp, "id", None)
            return resp
        except litellm.exceptions.AuthenticationError as e:
            e.message += " You can permanently set your API key with `mini-extra config set KEY VALUE`."
            raise e

    def query(self, messages: list[dict[str, str]], **kwargs) -> dict:
        for attempt in retry(logger=logger, abort_exceptions=self.abort_exceptions):
            with attempt:
                response = self._query(self._prepare_messages_for_api(messages), **kwargs)
        cost_output = self._calculate_cost(response)
        GLOBAL_MODEL_STATS.add(cost_output["cost"])
        message = response.model_dump() if hasattr(response, "model_dump") else dict(response)
        message["extra"] = {
            "actions": self._parse_actions(response),
            **cost_output,
            "timestamp": time.time(),
        }
        return message

    def _parse_actions(self, response) -> list[dict]:
        """Parse tool calls from the response API response."""
        tool_calls = []
        output = getattr(response, "output", [])
        for item in output:
            if getattr(item, "type", None) == "function_call":
                tool_calls.append(item)
        return parse_toolcall_actions(tool_calls, format_error_template=self.config.format_error_template)

    def format_observation_messages(
        self, message: dict, outputs: list[dict], template_vars: dict | None = None
    ) -> list[dict]:
        """Format execution outputs into tool result messages."""
        actions = message.get("extra", {}).get("actions", [])
        return format_toolcall_observation_messages(
            actions=actions,
            outputs=outputs,
            observation_template=self.config.observation_template,
            template_vars=template_vars,
            multimodal_regex=self.config.multimodal_regex,
        )

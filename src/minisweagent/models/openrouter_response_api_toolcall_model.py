import json
import logging
import os
import time

import requests

from minisweagent.models import GLOBAL_MODEL_STATS
from minisweagent.models.openrouter_model import (
    OpenRouterAPIError,
    OpenRouterAuthenticationError,
    OpenRouterModelConfig,
    OpenRouterRateLimitError,
)
from minisweagent.models.utils.actions_toolcall_response import (
    BASH_TOOL_RESPONSE_API,
    format_toolcall_observation_messages,
    parse_toolcall_actions_response,
)
from minisweagent.models.utils.retry import retry

logger = logging.getLogger("openrouter_response_api_toolcall_model")


class OpenRouterResponseAPIToolcallModelConfig(OpenRouterModelConfig):
    format_error_template: str = "{{ error }}"


class OpenRouterResponseAPIToolcallModel:
    """OpenRouter model using the Responses API with native tool calling.

    Note: OpenRouter's Responses API is stateless - each request must include
    the full conversation history. previous_response_id is not supported.
    See: https://openrouter.ai/docs/api/reference/responses/overview
    """

    abort_exceptions: list[type[Exception]] = [OpenRouterAuthenticationError, KeyboardInterrupt]

    def __init__(self, **kwargs):
        self.config = OpenRouterResponseAPIToolcallModelConfig(**kwargs)
        self._api_url = "https://openrouter.ai/api/v1/responses"
        self._api_key = os.getenv("OPENROUTER_API_KEY", "")

    def _query(self, messages: list[dict[str, str]], **kwargs):
        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.config.model_name,
            "input": messages,
            "tools": [BASH_TOOL_RESPONSE_API],
            **(self.config.model_kwargs | kwargs),
        }
        try:
            response = requests.post(self._api_url, headers=headers, data=json.dumps(payload), timeout=60)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.HTTPError as e:
            if response.status_code == 401:
                error_msg = "Authentication failed. You can permanently set your API key with `mini-extra config set OPENROUTER_API_KEY YOUR_KEY`."
                raise OpenRouterAuthenticationError(error_msg) from e
            elif response.status_code == 429:
                raise OpenRouterRateLimitError("Rate limit exceeded") from e
            else:
                raise OpenRouterAPIError(f"HTTP {response.status_code}: {response.text}") from e
        except requests.exceptions.RequestException as e:
            raise OpenRouterAPIError(f"Request failed: {e}") from e

    def _prepare_messages_for_api(self, messages: list[dict]) -> list[dict]:
        """Prepare messages for OpenRouter's stateless Responses API.

        Flattens response objects into their output items since OpenRouter
        doesn't support previous_response_id.
        """
        result = []
        for msg in messages:
            if msg.get("object") == "response":
                for item in msg.get("output", []):
                    result.append({k: v for k, v in item.items() if k != "extra"})
            else:
                result.append({k: v for k, v in msg.items() if k != "extra"})
        return result

    def query(self, messages: list[dict[str, str]], **kwargs) -> dict:
        for attempt in retry(logger=logger, abort_exceptions=self.abort_exceptions):
            with attempt:
                response = self._query(self._prepare_messages_for_api(messages), **kwargs)
        cost_output = self._calculate_cost(response)
        GLOBAL_MODEL_STATS.add(cost_output["cost"])
        message = dict(response)
        message["extra"] = {
            "actions": self._parse_actions(response),
            **cost_output,
            "timestamp": time.time(),
        }
        return message

    def _calculate_cost(self, response: dict) -> dict[str, float]:
        usage = response.get("usage", {})
        cost = usage.get("cost", 0.0)
        if cost <= 0.0 and self.config.cost_tracking != "ignore_errors":
            raise RuntimeError(
                f"No valid cost information available from OpenRouter API for model {self.config.model_name}: "
                f"Usage {usage}, cost {cost}. Cost must be > 0.0. Set cost_tracking: 'ignore_errors' in your config file or "
                "export MSWEA_COST_TRACKING='ignore_errors' to ignore cost tracking errors "
                "(for example for free/local models), more information at https://klieret.short.gy/mini-local-models "
                "for more details. Still stuck? Please open a github issue at https://github.com/SWE-agent/mini-swe-agent/issues/new/choose!"
            )
        return {"cost": cost}

    def _parse_actions(self, response: dict) -> list[dict]:
        return parse_toolcall_actions_response(
            response.get("output", []), format_error_template=self.config.format_error_template
        )

    def format_message(self, **kwargs) -> dict:
        role = kwargs.get("role", "user")
        content = kwargs.get("content", "")
        extra = kwargs.get("extra")
        content_items = [{"type": "input_text", "text": content}] if isinstance(content, str) else content
        msg = {"type": "message", "role": role, "content": content_items}
        if extra:
            msg["extra"] = extra
        return msg

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

    def get_template_vars(self, **kwargs) -> dict:
        return self.config.model_dump()

    def serialize(self) -> dict:
        return {
            "info": {
                "config": {
                    "model": self.config.model_dump(mode="json"),
                    "model_type": f"{self.__class__.__module__}.{self.__class__.__name__}",
                },
            }
        }

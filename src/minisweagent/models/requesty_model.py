import json
import logging
import os
import time
from typing import Any, Literal

import requests
from pydantic import BaseModel

from minisweagent.models import GLOBAL_MODEL_STATS
from minisweagent.models.utils.actions_text import format_observation_messages, parse_regex_actions
from minisweagent.models.utils.anthropic_utils import _reorder_anthropic_thinking_blocks
from minisweagent.models.utils.cache_control import set_cache_control
from minisweagent.models.utils.openai_multimodal import expand_multimodal_content
from minisweagent.models.utils.retry import retry

logger = logging.getLogger("requesty_model")


class RequestyModelConfig(BaseModel):
    model_name: str
    model_kwargs: dict[str, Any] = {}
    set_cache_control: Literal["default_end"] | None = None
    """Set explicit cache control markers, for example for Anthropic models"""
    action_regex: str = r"```mswea_bash_command\s*\n(.*?)\n```"
    """Regex to extract the action from the LM's output."""
    format_error_template: str = (
        "Please always provide EXACTLY ONE action in triple backticks, found {{actions|length}} actions."
    )
    """Template used when the LM's output is not in the expected format."""
    observation_template: str = (
        "{% if output.exception_info %}<exception>{{output.exception_info}}</exception>\n{% endif %}"
        "<returncode>{{output.returncode}}</returncode>\n<output>\n{{output.output}}</output>"
    )
    """Template used to render the observation after executing an action."""
    multimodal_regex: str = ""
    """Regex to extract multimodal content. Empty string disables multimodal processing."""


class RequestyAPIError(Exception):
    """Custom exception for Requesty API errors."""

    pass


class RequestyAuthenticationError(Exception):
    """Custom exception for Requesty authentication errors."""

    pass


class RequestyRateLimitError(Exception):
    """Custom exception for Requesty rate limit errors."""

    pass


class RequestyModel:
    abort_exceptions: list[type[Exception]] = [RequestyAuthenticationError, KeyboardInterrupt]

    def __init__(self, **kwargs):
        self.config = RequestyModelConfig(**kwargs)
        self._api_url = "https://router.requesty.ai/v1/chat/completions"
        self._api_key = os.getenv("REQUESTY_API_KEY", "")

    def _query(self, messages: list[dict[str, str]], **kwargs):
        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/SWE-agent/mini-swe-agent",
            "X-Title": "mini-swe-agent",
        }

        payload = {
            "model": self.config.model_name,
            "messages": messages,
            **(self.config.model_kwargs | kwargs),
        }

        try:
            response = requests.post(self._api_url, headers=headers, data=json.dumps(payload), timeout=60)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.HTTPError as e:
            if response.status_code == 401:
                error_msg = "Authentication failed. You can permanently set your API key with `mini-extra config set REQUESTY_API_KEY YOUR_KEY`."
                raise RequestyAuthenticationError(error_msg) from e
            elif response.status_code == 429:
                raise RequestyRateLimitError("Rate limit exceeded") from e
            else:
                raise RequestyAPIError(f"HTTP {response.status_code}: {response.text}") from e
        except requests.exceptions.RequestException as e:
            raise RequestyAPIError(f"Request failed: {e}") from e

    def _prepare_messages_for_api(self, messages: list[dict]) -> list[dict]:
        prepared = [{k: v for k, v in msg.items() if k != "extra"} for msg in messages]
        prepared = _reorder_anthropic_thinking_blocks(prepared)
        return set_cache_control(prepared, mode=self.config.set_cache_control)

    def query(self, messages: list[dict[str, str]], **kwargs) -> dict:
        for attempt in retry(logger=logger, abort_exceptions=self.abort_exceptions):
            with attempt:
                response = self._query(self._prepare_messages_for_api(messages), **kwargs)
        cost_output = self._calculate_cost(response)
        GLOBAL_MODEL_STATS.add(cost_output["cost"])
        message = dict(response["choices"][0]["message"])
        message["extra"] = {
            "actions": self._parse_actions(response),
            "response": response,
            **cost_output,
            "timestamp": time.time(),
        }
        return message

    def _calculate_cost(self, response) -> dict[str, float]:
        usage = response.get("usage", {})
        cost = usage.get("cost", 0.0)
        if cost == 0.0:
            raise RequestyAPIError(
                f"No cost information available from Requesty API for model {self.config.model_name}. "
                "Cost tracking is required but not provided by the API response."
            )
        return {"cost": cost}

    def _parse_actions(self, response: dict) -> list[dict]:
        """Parse actions from the model response. Raises FormatError if not exactly one action."""
        content = response["choices"][0]["message"]["content"] or ""
        return parse_regex_actions(
            content, action_regex=self.config.action_regex, format_error_template=self.config.format_error_template
        )

    def format_message(self, **kwargs) -> dict:
        return expand_multimodal_content(kwargs, pattern=self.config.multimodal_regex)

    def format_observation_messages(
        self, message: dict, outputs: list[dict], template_vars: dict | None = None
    ) -> list[dict]:
        """Format execution outputs into observation messages."""
        return format_observation_messages(
            outputs,
            observation_template=self.config.observation_template,
            template_vars=template_vars,
            multimodal_regex=self.config.multimodal_regex,
        )

    def get_template_vars(self, **kwargs) -> dict[str, Any]:
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

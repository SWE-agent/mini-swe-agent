import json
import logging
import os
import time
from typing import Any, Literal

import requests
from pydantic import BaseModel

from minisweagent import TextGenerationResult
from minisweagent.exceptions import ContextWindowExceeded, FormatError, is_context_window_error
from minisweagent.models import GLOBAL_MODEL_STATS
from minisweagent.models.utils.actions_toolcall import (
    BASH_TOOL,
    format_toolcall_observation_messages,
    parse_toolcall_actions,
)
from minisweagent.models.utils.anthropic_utils import _reorder_anthropic_thinking_blocks
from minisweagent.models.utils.cache_control import set_cache_control
from minisweagent.models.utils.openai_multimodal import expand_multimodal_content
from minisweagent.models.utils.retry import retry
from minisweagent.models.utils.text_generation import chat_text, request_kwargs, text_generation_result

logger = logging.getLogger("openrouter_model")


class OpenRouterModelConfig(BaseModel):
    model_name: str
    model_kwargs: dict[str, Any] = {}
    set_cache_control: Literal["default_end"] | None = None
    """Set explicit cache control markers, for example for Anthropic models"""
    cost_tracking: Literal["default", "ignore_errors"] = os.getenv("MSWEA_COST_TRACKING", "default")
    """Cost tracking mode for this model. Can be "default" or "ignore_errors" (ignore errors/missing cost info)"""
    format_error_template: str = "{{ error }}"
    """Template used when the LM's output is not in the expected format."""
    observation_template: str = (
        "{% if output.exception_info %}<exception>{{output.exception_info}}</exception>\n{% endif %}"
        "<returncode>{{output.returncode}}</returncode>\n<output>\n{{output.output}}</output>"
    )
    """Template used to render the observation after executing an action."""
    multimodal_regex: str = ""
    """Regex to extract multimodal content. Empty string disables multimodal processing."""


class OpenRouterAPIError(Exception):
    """Custom exception for OpenRouter API errors."""


class OpenRouterAuthenticationError(Exception):
    """Custom exception for OpenRouter authentication errors."""


class OpenRouterRateLimitError(Exception):
    """Custom exception for OpenRouter rate limit errors."""


class OpenRouterModel:
    abort_exceptions: list[type[Exception]] = [
        OpenRouterAuthenticationError,
        ContextWindowExceeded,
        KeyboardInterrupt,
    ]

    def __init__(self, **kwargs):
        self.config = OpenRouterModelConfig(**kwargs)
        self._api_url = "https://openrouter.ai/api/v1/chat/completions"
        self._api_key = os.getenv("OPENROUTER_API_KEY", "")

    def _query(self, messages: list[dict[str, str]], *, use_tools: bool = True, **kwargs):
        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }

        payload = {
            "model": self.config.model_name,
            "messages": messages,
            "usage": {"include": True},
            **request_kwargs(self.config.model_kwargs, kwargs, [BASH_TOOL] if use_tools else None),
        }

        try:
            response = requests.post(self._api_url, headers=headers, data=json.dumps(payload), timeout=60)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.HTTPError as e:
            self._raise_http_error(response, e)
        except requests.exceptions.RequestException as e:
            raise OpenRouterAPIError(f"Request failed: {e}") from e

    def _raise_http_error(self, response: requests.Response, error: Exception) -> None:
        if is_context_window_error(response.text):
            raise ContextWindowExceeded(response.text) from error
        if response.status_code == 401:
            raise OpenRouterAuthenticationError(
                "Authentication failed. You can permanently set your API key with "
                "`mini-extra config set OPENROUTER_API_KEY YOUR_KEY`."
            ) from error
        if response.status_code == 429:
            raise OpenRouterRateLimitError("Rate limit exceeded") from error
        raise OpenRouterAPIError(f"HTTP {response.status_code}: {response.text}") from error

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
        try:
            actions = self._parse_actions(response)
        except FormatError as e:
            e.messages[0]["extra"].update(cost_output)
            # dict(response) is a shallow copy; the original response object is not mutated.
            e.messages[0]["extra"]["response"] = dict(response)
            raise
        message = dict(response["choices"][0]["message"])
        message["extra"] = {
            "actions": actions,
            "response": response,
            **cost_output,
            "timestamp": time.time(),
        }
        return message

    def generate_text(self, messages: list[dict[str, str]], **kwargs) -> TextGenerationResult:
        for attempt in retry(logger=logger, abort_exceptions=self.abort_exceptions):
            with attempt:
                response = self._query(self._prepare_messages_for_api(messages), use_tools=False, **kwargs)
        cost_output = self._calculate_cost(response)
        GLOBAL_MODEL_STATS.add(cost_output["cost"])
        return text_generation_result(chat_text(response), response, cost_output["cost"])

    def _calculate_cost(self, response) -> dict[str, float]:
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
        """Parse tool calls from the response. Raises FormatError if unknown tool."""
        tool_calls = response["choices"][0]["message"].get("tool_calls") or []
        tool_calls = [_DictToObj(tc) for tc in tool_calls]
        return parse_toolcall_actions(
            tool_calls,
            format_error_template=self.config.format_error_template,
            template_kwargs={"finish_reason": response["choices"][0].get("finish_reason")},
        )

    def format_message(self, **kwargs) -> dict:
        return expand_multimodal_content(kwargs, pattern=self.config.multimodal_regex)

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


class _DictToObj:
    """Simple wrapper to convert dict to object with attribute access."""

    def __init__(self, d: dict):
        self._d = d
        self.id = d.get("id")
        self.function = _DictToObj(d.get("function", {})) if "function" in d else None
        self.name = d.get("name")
        self.arguments = d.get("arguments")

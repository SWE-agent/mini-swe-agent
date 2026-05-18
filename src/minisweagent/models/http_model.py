import json
import logging
import os
import time
from typing import Any, Literal

import requests
from pydantic import BaseModel

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

logger = logging.getLogger("http_model")


def _default_api_url() -> str:
    return os.getenv("LLM_API_URL", "https://api.openai.com/v1/chat/completions")


def _default_api_key_env() -> str:
    return "LLM_API_KEY" if os.getenv("LLM_API_KEY") else "OPENAI_API_KEY"


class HttpModelConfig(BaseModel):
    model_name: str
    """Model name as expected by the API (e.g., `gpt-4o`, `gemini-2.5-flash`).
    Litellm-style provider prefixes like `gemini/` are stripped automatically when strip_provider_prefix is True."""
    api_url: str = ""
    """OpenAI-compatible chat completions endpoint.
    Falls back to the LLM_API_URL environment variable, then https://api.openai.com/v1/chat/completions."""
    api_key_env: str = ""
    """Name of the environment variable holding the API key.
    Falls back to LLM_API_KEY if set in the environment, otherwise OPENAI_API_KEY."""
    strip_provider_prefix: bool = True
    """Strip litellm-style provider prefix from model_name (e.g., 'gemini/gemini-2.5-flash' → 'gemini-2.5-flash')."""
    ssl_verify: bool | str = True
    """SSL verification: True (default), False (disable), or a path to a CA bundle."""
    extra_headers: dict[str, str] = {}
    """Additional HTTP headers to include in the request."""
    model_kwargs: dict[str, Any] = {}
    """Additional arguments passed to the API payload."""
    set_cache_control: Literal["default_end"] | None = None
    """Set explicit cache control markers, for example for Anthropic models."""
    cost_tracking: Literal["default", "ignore_errors"] = os.getenv("MSWEA_COST_TRACKING", "default")
    """Cost tracking mode. Can be "default" or "ignore_errors"."""
    format_error_template: str = "{{ error }}"
    """Template used when the LM's output is not in the expected format."""
    observation_template: str = (
        "{% if output.exception_info %}<exception>{{output.exception_info}}</exception>\n{% endif %}"
        "<returncode>{{output.returncode}}</returncode>\n<output>\n{{output.output}}</output>"
    )
    """Template used to render the observation after executing an action."""
    multimodal_regex: str = ""
    """Regex to extract multimodal content. Empty string disables multimodal processing."""


class HttpAPIError(Exception):
    """HTTP API error."""


class HttpAuthenticationError(Exception):
    """HTTP API authentication error."""


class HttpRateLimitError(Exception):
    """HTTP API rate limit error."""


class HttpModel:
    abort_exceptions: list[type[Exception]] = [HttpAuthenticationError, KeyboardInterrupt]

    def __init__(self, **kwargs):
        self.config = HttpModelConfig(**kwargs)
        if not self.config.api_url:
            self.config.api_url = _default_api_url()
        if not self.config.api_key_env:
            self.config.api_key_env = _default_api_key_env()
        self._api_key = os.getenv(self.config.api_key_env, "")
        self._model_name = (
            self.config.model_name.split("/", 1)[-1] if self.config.strip_provider_prefix else self.config.model_name
        )

    def _query(self, messages: list[dict[str, str]], **kwargs):
        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
            **self.config.extra_headers,
        }

        payload = {
            "model": self._model_name,
            "messages": messages,
            "tools": [BASH_TOOL],
            **(self.config.model_kwargs | kwargs),
        }

        try:
            response = requests.post(
                self.config.api_url,
                headers=headers,
                data=json.dumps(payload),
                verify=self.config.ssl_verify,
                timeout=120,
            )
            response.raise_for_status()
            return response.json()
        except requests.exceptions.HTTPError as e:
            if response.status_code == 401:
                raise HttpAuthenticationError(
                    f"Authentication failed. Set your API key via the {self.config.api_key_env} environment variable."
                ) from e
            elif response.status_code == 429:
                raise HttpRateLimitError("Rate limit exceeded") from e
            else:
                raise HttpAPIError(f"HTTP {response.status_code}: {response.text}") from e
        except requests.exceptions.SSLError as e:
            raise HttpAPIError(
                f"SSL error: {e}. Set ssl_verify: false in your model config to disable SSL verification, "
                "or ssl_verify: '/path/to/ca-bundle.crt' to use a custom CA certificate."
            ) from e
        except requests.exceptions.RequestException as e:
            raise HttpAPIError(f"Request failed: {e}") from e

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
        if cost <= 0.0 and self.config.cost_tracking != "ignore_errors":
            raise RuntimeError(
                f"No valid cost information in API response for model {self.config.model_name}: "
                f"usage={usage}, cost={cost}. Set cost_tracking: 'ignore_errors' in your config or "
                "export MSWEA_COST_TRACKING='ignore_errors' to skip cost tracking."
            )
        return {"cost": cost}

    def _parse_actions(self, response: dict) -> list[dict]:
        """Parse tool calls from the response. Raises FormatError if unknown tool."""
        tool_calls = response["choices"][0]["message"].get("tool_calls") or []
        tool_calls = [_DictToObj(tc) for tc in tool_calls]
        return parse_toolcall_actions(tool_calls, format_error_template=self.config.format_error_template)

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

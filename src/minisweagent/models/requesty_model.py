import json
import logging
import os
import re
import time
from typing import Any

import requests
from jinja2 import StrictUndefined, Template
from pydantic import BaseModel
from tenacity import (
    before_sleep_log,
    retry,
    retry_if_not_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from minisweagent.exceptions import FormatError
from minisweagent.models import GLOBAL_MODEL_STATS

logger = logging.getLogger("requesty_model")


class RequestyModelConfig(BaseModel):
    model_name: str
    model_kwargs: dict[str, Any] = {}
    action_regex: str = r"```bash\s*\n(.*?)\n```"
    """Regex to extract the action from the LM's output."""
    format_error_template: str = (
        "Please always provide EXACTLY ONE action in triple backticks, found {{actions|length}} actions."
    )
    """Template used when the LM's output is not in the expected format."""


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
    def __init__(self, **kwargs):
        self.config = RequestyModelConfig(**kwargs)
        self.cost = 0.0
        self.n_calls = 0
        self._api_url = "https://router.requesty.ai/v1/chat/completions"
        self._api_key = os.getenv("REQUESTY_API_KEY", "")

    @retry(
        reraise=True,
        stop=stop_after_attempt(10),
        wait=wait_exponential(multiplier=1, min=4, max=60),
        before_sleep=before_sleep_log(logger, logging.WARNING),
        retry=retry_if_not_exception_type(
            (
                RequestyAuthenticationError,
                KeyboardInterrupt,
            )
        ),
    )
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

    def query(self, messages: list[dict[str, str]], **kwargs) -> list[dict]:
        response = self._query([{k: v for k, v in msg.items() if k != "extra"} for msg in messages], **kwargs)
        cost_output = self._calculate_cost(response)
        self.n_calls += 1
        self.cost += cost_output["cost"]
        GLOBAL_MODEL_STATS.add(cost_output["cost"])
        content = response["choices"][0]["message"]["content"] or ""
        return [
            {
                "role": "assistant",
                "content": content,
                "extra": {
                    "action": self.parse_action(content),
                    "response": response,
                    **cost_output,
                    "timestamp": time.time(),
                },
            }
        ]

    def _calculate_cost(self, response) -> dict[str, float]:
        usage = response.get("usage", {})
        cost = usage.get("cost", 0.0)
        if cost == 0.0:
            raise RequestyAPIError(
                f"No cost information available from Requesty API for model {self.config.model_name}. "
                "Cost tracking is required but not provided by the API response."
            )
        return {"cost": cost}

    def parse_action(self, content: str) -> str:
        """Parse the action from the model output. Raises FormatError if not exactly one action."""
        actions = re.findall(self.config.action_regex, content, re.DOTALL)
        if len(actions) != 1:
            raise FormatError(
                Template(self.config.format_error_template, undefined=StrictUndefined).render(actions=actions)
            )
        return actions[0].strip()

    def get_template_vars(self, **kwargs) -> dict[str, Any]:
        return self.config.model_dump() | {"n_model_calls": self.n_calls, "model_cost": self.cost}

    def serialize(self) -> dict:
        return {
            "info": {
                "model_stats": {
                    "instance_cost": self.cost,
                    "api_calls": self.n_calls,
                },
                "config": {
                    "model": self.config.model_dump(mode="json"),
                    "model_type": f"{self.__class__.__module__}.{self.__class__.__name__}",
                },
            }
        }

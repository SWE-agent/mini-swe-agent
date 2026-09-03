import json
import logging

import requests

from minisweagent.models.openrouter_model import (
    OpenRouterAPIError,
    OpenRouterModel,
    OpenRouterModelConfig,
)
from minisweagent.models.utils.actions_text import format_observation_messages, parse_regex_actions
from minisweagent.models.utils.text_generation import request_kwargs

logger = logging.getLogger("openrouter_textbased_model")


class OpenRouterTextbasedModelConfig(OpenRouterModelConfig):
    action_regex: str = r"```mswea_bash_command\s*\n(.*?)\n```"
    """Regex to extract the action from the LM's output."""
    format_error_template: str = (
        "Please always provide EXACTLY ONE action in triple backticks, found {{actions|length}} actions."
    )
    """Template used when the LM's output is not in the expected format."""


class OpenRouterTextbasedModel(OpenRouterModel):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.config = OpenRouterTextbasedModelConfig(**kwargs)

    def _query(self, messages: list[dict[str, str]], *, use_tools: bool = False, **kwargs):
        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }

        payload = {
            "model": self.config.model_name,
            "messages": messages,
            "usage": {"include": True},
            **request_kwargs(self.config.model_kwargs, kwargs, None),
        }

        try:
            response = requests.post(self._api_url, headers=headers, data=json.dumps(payload), timeout=60)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.HTTPError as e:
            self._raise_http_error(response, e)
        except requests.exceptions.RequestException as e:
            raise OpenRouterAPIError(f"Request failed: {e}") from e

    def _parse_actions(self, response: dict) -> list[dict]:
        """Parse actions from the model response. Raises FormatError if not exactly one action."""
        content = response["choices"][0]["message"]["content"] or ""
        return parse_regex_actions(
            content,
            action_regex=self.config.action_regex,
            format_error_template=self.config.format_error_template,
            template_kwargs={"finish_reason": response["choices"][0].get("finish_reason")},
        )

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

import json
import logging
import os
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any, Literal

import litellm
import openai
import requests
from litellm.litellm_core_utils.streaming_handler import CustomStreamWrapper
from litellm.types.utils import ChatCompletionMessageToolCall, Choices, Function, Message, ModelResponse
from pydantic import BaseModel

from minisweagent.exceptions import FormatError, ProviderTimeout
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

logger = logging.getLogger("litellm_model")


def _is_timeout_exception(e: Exception) -> bool:
    if isinstance(
        e,
        (
            TimeoutError,
            litellm.exceptions.Timeout,
            openai.APITimeoutError,
            requests.exceptions.Timeout,
        ),
    ):
        return True
    return any("timeout" in cls.__name__.lower() for cls in type(e).mro()) or "timed out" in str(e).lower()


class LitellmModelConfig(BaseModel):
    model_name: str
    """Model name. Highly recommended to include the provider in the model name, e.g., `anthropic/claude-sonnet-4-5-20250929`."""
    model_kwargs: dict[str, Any] = {}
    """Additional arguments passed to the API."""
    provider_timeout: float | None = 5.0
    """Default read timeout in seconds for streaming provider requests. Set to null to use the provider or LiteLLM default."""
    litellm_model_registry: Path | str | None = os.getenv("LITELLM_MODEL_REGISTRY_PATH")
    """Model registry for cost tracking and model metadata. See the local model guide (https://mini-swe-agent.com/latest/models/local_models/) for more details."""
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


class LitellmModel:
    abort_exceptions: list[type[Exception]] = [
        litellm.exceptions.UnsupportedParamsError,
        litellm.exceptions.NotFoundError,
        litellm.exceptions.PermissionDeniedError,
        litellm.exceptions.ContextWindowExceededError,
        litellm.exceptions.AuthenticationError,
        ProviderTimeout,
        KeyboardInterrupt,
    ]

    def __init__(self, *, config_class: Callable = LitellmModelConfig, **kwargs):
        self.config = config_class(**kwargs)
        if self.config.litellm_model_registry and Path(self.config.litellm_model_registry).is_file():
            litellm.utils.register_model(json.loads(Path(self.config.litellm_model_registry).read_text()))

    def _query(self, messages: list[dict[str, str]], **kwargs):
        try:
            response = litellm.completion(
                model=self.config.model_name,
                messages=messages,
                tools=[BASH_TOOL],
                **self._model_kwargs(**kwargs),
            )
            return self._consume_stream_response(response) if isinstance(response, CustomStreamWrapper) else response
        except litellm.exceptions.AuthenticationError as e:
            e.message += " You can permanently set your API key with `mini-extra config set KEY VALUE`."
            raise e
        except Exception as e:
            self._raise_provider_timeout(e)

    def _model_kwargs(self, **kwargs) -> dict[str, Any]:
        model_kwargs = self.config.model_kwargs | kwargs
        if (
            self.config.provider_timeout is not None
            and model_kwargs.get("stream") is True
            and not {"timeout", "request_timeout"} & model_kwargs.keys()
        ):
            model_kwargs["timeout"] = openai.Timeout(10.0, read=self.config.provider_timeout)
        return model_kwargs

    def _raise_provider_timeout(self, e: Exception) -> None:
        if not _is_timeout_exception(e):
            raise e
        raise ProviderTimeout(
            self.format_message(
                role="exit",
                content="ProviderTimeout: model provider did not respond within the configured timeout.",
                extra={"exit_status": "ProviderTimeout", "submission": "", "exception_str": str(e)},
            )
        ) from e

    def _consume_stream_response(self, stream: CustomStreamWrapper) -> ModelResponse:
        content: list[str] = []
        tool_calls_by_index: dict[int, dict[str, str]] = {}
        finish_reason = None
        response_id = None
        model = self.config.model_name
        created = None
        usage = None
        for chunk in stream:
            response_id = getattr(chunk, "id", response_id)
            model = getattr(chunk, "model", model)
            created = getattr(chunk, "created", created)
            usage = getattr(chunk, "usage", None) or usage
            if not chunk.choices:
                continue
            choice = chunk.choices[0]
            finish_reason = choice.finish_reason or finish_reason
            delta = choice.delta
            if delta.content:
                content.append(delta.content)
            for tool_call in delta.tool_calls or []:
                index = int(tool_call.index or 0)
                state = tool_calls_by_index.setdefault(
                    index, {"id": "", "type": "function", "name": "", "arguments": ""}
                )
                if tool_call.id:
                    state["id"] = tool_call.id
                if tool_call.type:
                    state["type"] = tool_call.type
                if tool_call.function:
                    if tool_call.function.name:
                        state["name"] = tool_call.function.name
                    if tool_call.function.arguments:
                        state["arguments"] += tool_call.function.arguments
        tool_calls = [
            ChatCompletionMessageToolCall(
                id=state["id"],
                type=state["type"],
                function=Function(name=state["name"], arguments=state["arguments"]),
            )
            for _, state in sorted(tool_calls_by_index.items())
        ]
        response = ModelResponse(
            id=response_id,
            created=created,
            model=model,
            usage=usage,
            choices=[
                Choices(
                    finish_reason=finish_reason,
                    message=Message(content="".join(content) or None, tool_calls=tool_calls or None),
                )
            ],
        )
        if usage is None:
            response._mswea_streamed_without_usage = True
        return response

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
        # Note: all model.query() implementations must persist the response on FormatError.
        try:
            actions = self._parse_actions(response)
        except FormatError as e:
            try:
                e.messages[0]["extra"]["response"] = response.model_dump(mode="json")
            except Exception:
                # model_dump failed (e.g. unserializable object); fall back to repr
                # so the spec contract ("response MUST be persisted") holds unconditionally.
                e.messages[0]["extra"]["response"] = repr(response)
            raise
        message = response.choices[0].message.model_dump()
        message["extra"] = {
            "actions": actions,
            "response": response.model_dump(),
            **cost_output,
            "timestamp": time.time(),
        }
        return message

    def _calculate_cost(self, response) -> dict[str, float]:
        if getattr(response, "_mswea_streamed_without_usage", False) is True:
            return {"cost": 0.0}
        try:
            cost = litellm.cost_calculator.completion_cost(response, model=self.config.model_name)
            if cost <= 0.0:
                raise ValueError(f"Cost must be > 0.0, got {cost}")
        except Exception as e:
            cost = 0.0
            if self.config.cost_tracking != "ignore_errors":
                msg = (
                    f"Error calculating cost for model {self.config.model_name}: {e}, perhaps it's not registered? "
                    "You can ignore this issue from your config file with cost_tracking: 'ignore_errors' or "
                    "globally with export MSWEA_COST_TRACKING='ignore_errors'. "
                    "Alternatively check the 'Cost tracking' section in the documentation at "
                    "https://klieret.short.gy/mini-local-models. "
                    " Still stuck? Please open a github issue at https://github.com/SWE-agent/mini-swe-agent/issues/new/choose!"
                )
                logger.critical(msg)
                raise RuntimeError(msg) from e
        return {"cost": cost}

    def _parse_actions(self, response) -> list[dict]:
        """Parse tool calls from the response. Raises FormatError if unknown tool."""
        tool_calls = response.choices[0].message.tool_calls or []
        return parse_toolcall_actions(
            tool_calls,
            format_error_template=self.config.format_error_template,
            template_kwargs={"finish_reason": response.choices[0].finish_reason},
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

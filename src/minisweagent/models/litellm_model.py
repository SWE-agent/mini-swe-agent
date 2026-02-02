import json
import logging
import os
import re
import time
from collections import Counter
from collections.abc import Callable
from pathlib import Path
from typing import Any, Literal

import litellm
from litellm.types.utils import Choices, Message, ModelResponse, Usage
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

logger = logging.getLogger("litellm_model")


class LitellmModelConfig(BaseModel):
    model_name: str
    """Model name. Highly recommended to include the provider in the model name, e.g., `anthropic/claude-sonnet-4-5-20250929`."""
    model_kwargs: dict[str, Any] = {}
    """Additional arguments passed to the API."""
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
    use_streaming: bool = os.getenv("MSWEA_USE_STREAMING", "false").lower() == "true"
    """Use streaming mode to avoid HTTP read timeouts on long generations. Default: false.
    When enabled, responses are streamed token-by-token, keeping the connection alive.
    This prevents timeout errors when vLLM takes >10 minutes to generate a response."""
    stream_include_usage: bool = os.getenv("MSWEA_STREAM_INCLUDE_USAGE", "true").lower() == "true"
    """Request usage stats in streaming responses when supported by the backend."""
    stream_guard_enabled: bool = os.getenv("MSWEA_STREAM_GUARD_ENABLED", "true").lower() == "true"
    """Enable client-side guard for pathological streaming repetition."""
    stream_guard_window: int = int(os.getenv("MSWEA_STREAM_GUARD_WINDOW", "8192"))
    """Rolling window size (chars) to scan for repeated closing-tag output."""
    stream_guard_tag_threshold: int = int(os.getenv("MSWEA_STREAM_GUARD_TAG_THRESHOLD", "50"))
    """Repetition threshold for closing-tag patterns before truncation."""


class LitellmModel:
    abort_exceptions: list[type[Exception]] = [
        litellm.exceptions.UnsupportedParamsError,
        litellm.exceptions.NotFoundError,
        litellm.exceptions.PermissionDeniedError,
        litellm.exceptions.ContextWindowExceededError,
        litellm.exceptions.AuthenticationError,
        KeyboardInterrupt,
    ]

    def __init__(self, *, config_class: Callable = LitellmModelConfig, **kwargs):
        self.config = config_class(**kwargs)
        if self.config.litellm_model_registry and Path(self.config.litellm_model_registry).is_file():
            litellm.utils.register_model(json.loads(Path(self.config.litellm_model_registry).read_text()))

    @staticmethod
    def _usage_from_response(response) -> Usage | None:
        if isinstance(response, dict):
            usage = response.get("usage")
        else:
            usage = getattr(response, "usage", None)
        if usage is None:
            return None
        if isinstance(usage, Usage):
            return usage
        if isinstance(usage, dict):
            return Usage(
                prompt_tokens=int(usage.get("prompt_tokens", 0) or 0),
                completion_tokens=int(usage.get("completion_tokens", 0) or 0),
                total_tokens=int(usage.get("total_tokens", 0) or 0),
            )
        prompt_tokens = getattr(usage, "prompt_tokens", None)
        completion_tokens = getattr(usage, "completion_tokens", None)
        total_tokens = getattr(usage, "total_tokens", None)
        if prompt_tokens is None and completion_tokens is None and total_tokens is None:
            return None
        return Usage(
            prompt_tokens=int(prompt_tokens or 0),
            completion_tokens=int(completion_tokens or 0),
            total_tokens=int(total_tokens or 0),
        )

    @staticmethod
    def _usage_is_valid(usage: Usage | None) -> bool:
        if usage is None:
            return False
        prompt_tokens = int(getattr(usage, "prompt_tokens", 0) or 0)
        total_tokens = int(getattr(usage, "total_tokens", 0) or 0)
        if prompt_tokens <= 0:
            return False
        if total_tokens < prompt_tokens:
            return False
        return True

    def _reconstruct_response_from_stream(self, stream_response) -> ModelResponse:
        """Accumulate streaming chunks into a complete ModelResponse."""
        content_parts = []
        tail_buffer = ""
        last_chunk = None
        usage = None

        for chunk in stream_response:
            last_chunk = chunk
            if chunk.choices and chunk.choices[0].delta.content:
                new_content = chunk.choices[0].delta.content
                content_parts.append(new_content)
                if self.config.stream_guard_enabled and new_content:
                    tail_buffer = (tail_buffer + new_content)[-self.config.stream_guard_window :]
                    if self._stream_guard_triggered(tail_buffer):
                        logger.warning(
                            "Streaming guard triggered; truncating response to avoid pathological repetition."
                        )
                        break
            chunk_usage = None
            if isinstance(chunk, dict):
                chunk_usage = chunk.get("usage")
            else:
                chunk_usage = getattr(chunk, "usage", None)
            if chunk_usage:
                usage = chunk_usage

        content = "".join(content_parts)

        usage_obj = None
        if usage:
            if isinstance(usage, Usage):
                usage_obj = usage
            elif isinstance(usage, dict):
                usage_obj = Usage(
                    prompt_tokens=int(usage.get("prompt_tokens", 0) or 0),
                    completion_tokens=int(usage.get("completion_tokens", 0) or 0),
                    total_tokens=int(usage.get("total_tokens", 0) or 0),
                )

        return ModelResponse(
            id=last_chunk.id if last_chunk else "stream-response",
            created=last_chunk.created if last_chunk else 0,
            model=last_chunk.model if last_chunk else self.config.model_name,
            choices=[
                Choices(
                    index=0,
                    finish_reason="stop",
                    message=Message(role="assistant", content=content),
                )
            ],
            usage=usage_obj
            or Usage(
                prompt_tokens=0,
                completion_tokens=len(content_parts),
                total_tokens=len(content_parts),
            ),
        )

    def _stream_guard_triggered(self, tail_buffer: str) -> bool:
        if not tail_buffer:
            return False
        tags = re.findall(r"</[A-Za-z0-9_.:-]+>", tail_buffer)
        if len(tags) < self.config.stream_guard_tag_threshold:
            return False
        counts = Counter(tags)
        return max(counts.values(), default=0) >= self.config.stream_guard_tag_threshold

    def _query(self, messages: list[dict[str, str]], **kwargs):
        try:
            if self.config.use_streaming:
                stream_kwargs = {}
                if self.config.stream_include_usage:
                    stream_kwargs["stream_options"] = {"include_usage": True}
                stream_response = litellm.completion(
                    model=self.config.model_name,
                    messages=messages,
                    tools=[BASH_TOOL],
                    stream=True,
                    **(self.config.model_kwargs | kwargs | stream_kwargs),
                )
                response = self._reconstruct_response_from_stream(stream_response)
                usage = self._usage_from_response(response)
                if not self._usage_is_valid(usage):
                    return litellm.completion(
                        model=self.config.model_name,
                        messages=messages,
                        tools=[BASH_TOOL],
                        **(self.config.model_kwargs | kwargs),
                    )
                return response
            return litellm.completion(
                model=self.config.model_name,
                messages=messages,
                tools=[BASH_TOOL],
                **(self.config.model_kwargs | kwargs),
            )
        except litellm.exceptions.AuthenticationError as e:
            e.message += " You can permanently set your API key with `mini-extra config set KEY VALUE`."
            raise e

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
        message = response.choices[0].message.model_dump()
        message["extra"] = {
            "actions": self._parse_actions(response),
            "response": response.model_dump(),
            **cost_output,
            "timestamp": time.time(),
        }
        return message

    def _calculate_cost(self, response) -> dict[str, float]:
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

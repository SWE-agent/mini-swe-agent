import json
import logging
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal

import litellm
from tenacity import (
    before_sleep_log,
    retry,
    retry_if_not_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from minisweagent.models import GLOBAL_MODEL_STATS
from minisweagent.models.utils.cache_control import set_cache_control

logger = logging.getLogger("portkey_model")

try:
    from portkey_ai import Portkey
except ImportError:
    raise ImportError(
        "The portkey-ai package is required to use PortkeyModel. Please install it with: pip install portkey-ai"
    )


@dataclass
class PortkeyModelConfig:
    model_name: str
    model_kwargs: dict[str, Any] = field(default_factory=dict)
    litellm_model_registry: Path | str | None = os.getenv("LITELLM_MODEL_REGISTRY_PATH")
    """We currently use litellm to calculate costs. Here you can register additional models to litellm's model registry.
    Note that this might change if we get better support for Portkey and change how we calculate costs.
    """
    litellm_model_name_override: str = ""
    """We currently use litellm to calculate costs. Here you can override the model name to use for litellm in case it
    doesn't match the Portkey model name.
    Note that this might change if we get better support for Portkey and change how we calculate costs.
    """
    set_cache_control: Literal["default_end"] | None = None
    """Set explicit cache control markers, for example for Anthropic models"""
    cost_tracking: Literal["default", "ignore_errors"] = os.getenv("MSWEA_COST_TRACKING", "default")
    """Cost tracking mode for this model. Can be "default" or "ignore_errors" (ignore errors/missing cost info)"""


class PortkeyModel:
    def __init__(self, *, config_class: type = PortkeyModelConfig, **kwargs):
        self.config = config_class(**kwargs)
        self.cost = 0.0
        self.n_calls = 0
        if self.config.litellm_model_registry and Path(self.config.litellm_model_registry).is_file():
            litellm.utils.register_model(json.loads(Path(self.config.litellm_model_registry).read_text()))

        # Get API key from environment or raise error
        self._api_key = os.getenv("PORTKEY_API_KEY")
        if not self._api_key:
            raise ValueError(
                "Portkey API key is required. Set it via the "
                "PORTKEY_API_KEY environment variable. You can permanently set it with "
                "`mini-extra config set PORTKEY_API_KEY YOUR_KEY`."
            )

        # Get virtual key from environment
        virtual_key = os.getenv("PORTKEY_VIRTUAL_KEY")

        # Initialize Portkey client
        client_kwargs = {"api_key": self._api_key}
        if virtual_key:
            client_kwargs["virtual_key"] = virtual_key

        self.client = Portkey(**client_kwargs)

    @retry(
        stop=stop_after_attempt(int(os.getenv("MSWEA_MODEL_RETRY_STOP_AFTER_ATTEMPT", "10"))),
        wait=wait_exponential(multiplier=1, min=4, max=60),
        before_sleep=before_sleep_log(logger, logging.WARNING),
        retry=retry_if_not_exception_type((KeyboardInterrupt, TypeError, ValueError)),
    )
    def _query(self, messages: list[dict[str, str]], **kwargs):
        # return self.client.with_options(metadata={"request_id": request_id}).chat.completions.create(
        return self.client.chat.completions.create(
            model=self.config.model_name,
            messages=messages,
            **(self.config.model_kwargs | kwargs),
        )

    def query(self, messages: list[dict[str, str]], **kwargs) -> dict:
        if self.config.set_cache_control:
            messages = set_cache_control(messages, mode=self.config.set_cache_control)
        response = self._query(messages, **kwargs)
        cost = self._calculate_cost(response)
        self.n_calls += 1
        self.cost += cost
        GLOBAL_MODEL_STATS.add(cost)
        return {
            "content": response.choices[0].message.content or "",
            "extra": {
                "response": response.model_dump(),
                "cost": cost,
            },
        }

    def get_template_vars(self) -> dict[str, Any]:
        return asdict(self.config) | {"n_model_calls": self.n_calls, "model_cost": self.cost}

    def _calculate_cost(self, response) -> float:
        response_for_cost_calc = response.model_copy()
        if self.config.litellm_model_name_override:
            if response_for_cost_calc.model:
                response_for_cost_calc.model = self.config.litellm_model_name_override
        prompt_tokens = response_for_cost_calc.usage.prompt_tokens
        if prompt_tokens is None:
            logger.warning(
                f"Prompt tokens are None for model {self.config.model_name}. Setting to 0. Full response: {response_for_cost_calc.model_dump()}"
            )
            prompt_tokens = 0
        total_tokens = response_for_cost_calc.usage.total_tokens
        completion_tokens = response_for_cost_calc.usage.completion_tokens
        if completion_tokens is None:
            logger.warning(
                f"Completion tokens are None for model {self.config.model_name}. Setting to 0. Full response: {response_for_cost_calc.model_dump()}"
            )
            completion_tokens = 0
        if total_tokens - prompt_tokens - completion_tokens != 0:
            # This is most likely related to how portkey treats cached tokens: It doesn't count them towards the prompt tokens (?)
            logger.warning(
                f"WARNING: Total tokens - prompt tokens - completion tokens != 0: {response_for_cost_calc.model_dump()}."
                " This is probably a portkey bug or incompatibility with litellm cost tracking. "
                "Setting prompt tokens based on total tokens and completion tokens. You might want to double check your costs. "
                f"Full response: {response_for_cost_calc.model_dump()}"
            )
            response_for_cost_calc.usage.prompt_tokens = total_tokens - completion_tokens
        try:
            cost = litellm.cost_calculator.completion_cost(
                response_for_cost_calc, model=self.config.litellm_model_name_override or None
            )
            assert cost >= 0.0, f"Cost is negative: {cost}"
        except Exception as e:
            cost = 0.0
            if self.config.cost_tracking != "ignore_errors":
                msg = (
                    f"Error calculating cost for model {self.config.model_name} based on {response_for_cost_calc.model_dump()}: {e}. "
                    "You can ignore this issue from your config file with cost_tracking: 'ignore_errors' or "
                    "globally with export MSWEA_COST_TRACKING='ignore_errors' to ignore this error. "
                    "Alternatively check the 'Cost tracking' section in the documentation at "
                    "https://klieret.short.gy/mini-local-models. "
                    "Still stuck? Please open a github issue at https://github.com/SWE-agent/mini-swe-agent/issues/new/choose!"
                )
                logger.critical(msg)
                raise RuntimeError(msg) from e
        return cost

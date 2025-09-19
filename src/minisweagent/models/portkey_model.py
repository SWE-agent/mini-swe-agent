import logging
import os
import time
import uuid
from dataclasses import asdict, dataclass, field
from typing import Any

import requests
from tenacity import (
    before_sleep_log,
    retry,
    retry_if_not_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from minisweagent.models import GLOBAL_MODEL_STATS

logger = logging.getLogger("portkey_model")

try:
    from portkey_ai import Portkey
except ImportError:
    Portkey = None


@dataclass
class PortkeyModelConfig:
    model_name: str
    model_kwargs: dict[str, Any] = field(default_factory=dict)


class PortkeyModel:
    def __init__(self, **kwargs):
        if Portkey is None:
            raise ImportError(
                "The portkey-ai package is required to use PortkeyModel. Please install it with: pip install portkey-ai"
            )
        self.config = PortkeyModelConfig(**kwargs)
        self.cost = 0.0
        self.n_calls = 0

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

    def _generate_request_id(self) -> str:
        """Generate a unique request ID for tracking."""
        return f"mini-swe-{uuid.uuid4().hex[:8]}-{int(time.time())}"

    @retry(
        stop=stop_after_attempt(10),
        wait=wait_exponential(multiplier=1, min=4, max=60),
        before_sleep=before_sleep_log(logger, logging.WARNING),
        retry=retry_if_not_exception_type((KeyboardInterrupt,)),
    )
    def _query(self, messages: list[dict[str, str]], request_id: str, **kwargs):
        return self.client.with_options(metadata={"request_id": request_id}).chat.completions.create(
            model=self.config.model_name,
            messages=messages,
            **(self.config.model_kwargs | kwargs),
        )

    def _get_cost_from_analytics(self, request_id: str) -> float:
        """Retrieve cost information from Portkey analytics API for a specific request_id."""
        if not self._api_key:
            return 0.0

        # Query Portkey analytics API for cost data by request_id
        url = "https://api.portkey.ai/v1/analytics/groups/metadata/request_id"
        headers = {"x-portkey-api-key": self._api_key, "Content-Type": "application/json"}

        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()

        analytics_data = response.json()

        # Find the specific request_id in the analytics data
        for group in analytics_data.get("groups", []):
            if group.get("metadata", {}).get("request_id") == request_id:
                return group["cost"]

        raise RuntimeError(f"No cost data found for request_id: {request_id}")

    def query(self, messages: list[dict[str, str]], **kwargs) -> dict:
        request_id = self._generate_request_id()
        response = self._query(messages, request_id, **kwargs)
        cost = self._get_cost_from_analytics(request_id)

        self.n_calls += 1
        self.cost += cost
        GLOBAL_MODEL_STATS.add(cost)

        return {
            "content": response.choices[0].message.content or "",
            "extra": {
                "response": response.model_dump() if hasattr(response, "model_dump") else str(response),
                "request_id": request_id,
                "cost": cost,
            },
        }

    def get_template_vars(self) -> dict[str, Any]:
        return asdict(self.config) | {"n_model_calls": self.n_calls, "model_cost": self.cost}

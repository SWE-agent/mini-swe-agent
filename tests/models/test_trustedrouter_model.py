import json
from unittest.mock import MagicMock, patch

import pytest

from minisweagent.models.trustedrouter_model import TrustedRouterAPIError, TrustedRouterModel


def _response(usage: dict) -> dict:
    tool_call = {"id": "call_1", "type": "function", "function": {"name": "bash", "arguments": '{"command": "ls"}'}}
    return {
        "choices": [{"message": {"role": "assistant", "content": None, "tool_calls": [tool_call]}}],
        "usage": usage,
    }


def test_query_posts_to_trustedrouter_and_reads_microdollar_cost(monkeypatch):
    monkeypatch.setenv("TRUSTEDROUTER_API_KEY", "test-key")
    http_response = MagicMock()
    http_response.json.return_value = _response({"cost_microdollars": 1200, "total_cost_microdollars": 1500})
    with patch("minisweagent.models.trustedrouter_model.requests.post", return_value=http_response) as post:
        message = TrustedRouterModel(model_name="trustedrouter/auto").query([{"role": "user", "content": "hi"}])

    assert post.call_args.args[0] == "https://api.trustedrouter.com/v1/chat/completions"
    assert post.call_args.kwargs["headers"]["Authorization"] == "Bearer test-key"
    assert json.loads(post.call_args.kwargs["data"])["model"] == "trustedrouter/auto"
    assert message["extra"]["cost"] == pytest.approx(0.0015)
    assert message["extra"]["actions"] == [{"command": "ls", "tool_call_id": "call_1"}]


def test_calculate_cost_falls_back_to_cost_microdollars():
    model = TrustedRouterModel(model_name="trustedrouter/auto")
    assert model._calculate_cost(_response({"cost_microdollars": 2500})) == {"cost": pytest.approx(0.0025)}


def test_calculate_cost_missing_raises_unless_ignored():
    with pytest.raises(TrustedRouterAPIError, match="No cost information"):
        TrustedRouterModel(model_name="trustedrouter/auto")._calculate_cost(_response({}))
    model = TrustedRouterModel(model_name="trustedrouter/auto", cost_tracking="ignore_errors")
    assert model._calculate_cost(_response({})) == {"cost": 0.0}

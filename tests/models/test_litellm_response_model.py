from unittest.mock import patch

import openai
import pytest

from minisweagent.exceptions import ProviderTimeout
from minisweagent.models.litellm_response_model import LitellmResponseModel


def test_query_includes_default_provider_timeout():
    model = LitellmResponseModel(model_name="gpt-4", model_kwargs={"stream": True})

    with patch("minisweagent.models.litellm_response_model.litellm.responses") as mock_responses:
        model._query([{"role": "user", "content": "test"}])
        assert isinstance(mock_responses.call_args.kwargs["timeout"], openai.Timeout)
        assert mock_responses.call_args.kwargs["timeout"].read == 5.0


def test_query_timeout_raises_provider_timeout():
    model = LitellmResponseModel(model_name="gpt-4")

    with patch(
        "minisweagent.models.litellm_response_model.litellm.responses",
        side_effect=TimeoutError("request timed out"),
    ):
        with pytest.raises(ProviderTimeout) as exc_info:
            model._query([{"role": "user", "content": "test"}])
    assert exc_info.value.messages[0]["extra"]["exit_status"] == "ProviderTimeout"

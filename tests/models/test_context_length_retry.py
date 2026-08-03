from unittest.mock import Mock, patch

import pytest
import requests

from minisweagent.models.openrouter_model import OpenRouterContextLengthError, OpenRouterModel
from minisweagent.models.openrouter_response_model import OpenRouterResponseModel
from minisweagent.models.openrouter_textbased_model import OpenRouterTextbasedModel
from minisweagent.models.requesty_model import RequestyContextLengthError, RequestyModel


@pytest.mark.parametrize(
    ("model_class", "error_class"),
    [
        (OpenRouterModel, OpenRouterContextLengthError),
        (OpenRouterTextbasedModel, OpenRouterContextLengthError),
        (OpenRouterResponseModel, OpenRouterContextLengthError),
        (RequestyModel, RequestyContextLengthError),
    ],
)
def test_context_length_errors_abort_without_retry(model_class, error_class, monkeypatch):
    monkeypatch.setenv("MSWEA_MODEL_RETRY_STOP_AFTER_ATTEMPT", "3")
    model = model_class(model_name="test/model")
    response = Mock(status_code=400, text='{"error":{"code":"context_length_exceeded"}}')
    response.raise_for_status.side_effect = requests.exceptions.HTTPError()

    with patch("requests.post", return_value=response) as post:
        with pytest.raises(error_class):
            model.query([{"role": "user", "content": "x"}])

    assert post.call_count == 1

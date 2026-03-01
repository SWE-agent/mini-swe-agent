from unittest.mock import patch

import pytest

from minisweagent.models.litellm_response_model import LitellmResponseModel


def _missing_module_error(module_name: str) -> ModuleNotFoundError:
    exc = ModuleNotFoundError(f"No module named '{module_name}'")
    exc.name = module_name
    return exc


def test_bedrock_missing_boto3_gets_actionable_error():
    model = LitellmResponseModel(model_name="bedrock/us-east-1/test-model")

    with patch("minisweagent.models.litellm_response_model.litellm.responses") as mock_responses:
        mock_responses.side_effect = _missing_module_error("boto3")

        with pytest.raises(RuntimeError) as exc_info:
            model.query([{"role": "user", "content": "test"}])

    message = str(exc_info.value)
    assert "Bedrock models require the AWS SDK" in message
    assert "AWS_BEARER_TOKEN_BEDROCK" in message

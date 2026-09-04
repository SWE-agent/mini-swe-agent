import os
from unittest.mock import patch

import pytest

from minisweagent.models import get_model
from minisweagent.models.atlascloud_model import AtlasCloudModel


def test_atlascloud_model_configures_litellm_openai_compatible_kwargs():
    with patch.dict(os.environ, {"ATLASCLOUD_API_KEY": "test-key"}, clear=True):
        model = AtlasCloudModel(model_name="atlascloud/qwen/qwen3.5-flash", model_kwargs={"temperature": 0.2})

    assert model.config.model_name == "openai/qwen/qwen3.5-flash"
    assert model.config.model_kwargs["api_key"] == "test-key"
    assert model.config.model_kwargs["api_base"] == "https://api.atlascloud.ai/v1"
    assert model.config.model_kwargs["temperature"] == 0.2
    assert model.config.cost_tracking == "ignore_errors"


def test_atlascloud_model_uses_custom_api_base_and_key_from_config():
    with patch.dict(os.environ, {}, clear=True):
        model = AtlasCloudModel(
            model_name="atlascloud/deepseek-ai/deepseek-v4-pro",
            model_kwargs={"api_key": "config-key", "api_base": "https://example.test/v1"},
        )

    assert model.config.model_name == "openai/deepseek-ai/deepseek-v4-pro"
    assert model.config.model_kwargs["api_key"] == "config-key"
    assert model.config.model_kwargs["api_base"] == "https://example.test/v1"


def test_atlascloud_model_requires_api_key():
    with patch.dict(os.environ, {}, clear=True):
        with pytest.raises(ValueError, match="ATLASCLOUD_API_KEY"):
            AtlasCloudModel(model_name="atlascloud/qwen/qwen3.5-flash")


def test_atlascloud_query_passes_openai_compatible_kwargs_to_litellm():
    with patch.dict(os.environ, {"ATLASCLOUD_API_KEY": "test-key"}, clear=True):
        model = AtlasCloudModel(model_name="atlascloud/qwen/qwen3.5-flash")

    with patch("minisweagent.models.litellm_model.litellm.completion") as mock_completion:
        model._query([{"role": "user", "content": "hello"}], max_tokens=32)

    call_kwargs = mock_completion.call_args.kwargs
    assert call_kwargs["model"] == "openai/qwen/qwen3.5-flash"
    assert call_kwargs["api_key"] == "test-key"
    assert call_kwargs["api_base"] == "https://api.atlascloud.ai/v1"
    assert call_kwargs["max_tokens"] == 32


def test_get_model_selects_atlascloud_adapter_for_prefix():
    with patch.dict(os.environ, {"ATLASCLOUD_API_KEY": "test-key"}, clear=True):
        model = get_model("atlascloud/qwen/qwen3.5-flash")

    assert isinstance(model, AtlasCloudModel)

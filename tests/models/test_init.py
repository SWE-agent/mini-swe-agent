import os
from unittest.mock import patch

from microsweagent.models import get_model, get_model_class, get_model_name
from microsweagent.models.test_models import DeterministicModel


class TestGetModelName:
    def test_input_model_name_takes_precedence(self):
        """Test that explicit input_model_name overrides all other sources."""
        with patch.dict(os.environ, {"MSWEA_MODEL_NAME": "env-model"}):
            assert get_model_name("input-model", {"model": {"model_name": "config-model"}}) == "input-model"

    def test_env_var_fallback(self):
        """Test that environment variable is used when no input provided."""
        with patch.dict(os.environ, {"MSWEA_MODEL_NAME": "env-model"}):
            assert get_model_name(None, {"model": {"model_name": "config-model"}}) == "env-model"

    def test_config_fallback(self):
        """Test that config model name is used when input and env are missing."""
        with patch.dict(os.environ, {}, clear=True):
            with patch("microsweagent.models.prompt_for_model_name", return_value="prompted-model"):
                assert get_model_name(None, {"model": {"model_name": "config-model"}}) == "config-model"

    def test_prompt_fallback(self):
        """Test that prompt is used when all other sources are missing."""
        with patch.dict(os.environ, {}, clear=True):
            with patch("microsweagent.models.prompt_for_model_name", return_value="prompted-model"):
                assert get_model_name(None, {}) == "prompted-model"
                assert get_model_name(None, None) == "prompted-model"


class TestGetModelClass:
    def test_anthropic_model_selection(self):
        """Test that anthropic-related model names return AnthropicModel."""
        from microsweagent.models.anthropic import AnthropicModel

        for name in ["anthropic", "sonnet", "opus", "claude-sonnet", "claude-opus"]:
            assert get_model_class(name) == AnthropicModel

    def test_litellm_model_fallback(self):
        """Test that non-anthropic model names return LitellmModel."""
        from microsweagent.models.litellm_model import LitellmModel

        for name in ["gpt-4", "gpt-3.5-turbo", "llama2", "random-model"]:
            assert get_model_class(name) == LitellmModel

    def test_partial_matches(self):
        """Test that partial string matches work correctly."""
        from microsweagent.models.anthropic import AnthropicModel
        from microsweagent.models.litellm_model import LitellmModel

        assert get_model_class("my-anthropic-model") == AnthropicModel
        assert get_model_class("sonnet-latest") == AnthropicModel
        assert get_model_class("opus-v2") == AnthropicModel
        assert get_model_class("gpt-anthropic-style") == AnthropicModel
        assert get_model_class("totally-different") == LitellmModel


class TestGetModel:
    def test_config_deep_copy(self):
        """Test that get_model preserves original config via deep copy."""
        original_config = {"model_kwargs": {"api_key": "original"}, "outputs": ["test"]}

        with patch("microsweagent.models.get_model_class") as mock_get_class:
            mock_get_class.return_value = lambda **kwargs: DeterministicModel(outputs=["test"], model_name="test")
            get_model("test-model", original_config)
            assert original_config["model_kwargs"]["api_key"] == "original"
            assert "model_name" not in original_config

    def test_integration_with_compatible_model(self):
        """Test get_model works end-to-end with a model that handles extra kwargs."""
        with patch("microsweagent.models.get_model_class") as mock_get_class:

            def compatible_model(**kwargs):
                # Filter to only what DeterministicModel accepts, provide defaults
                config_args = {k: v for k, v in kwargs.items() if k in ["outputs", "model_name"]}
                if "outputs" not in config_args:
                    config_args["outputs"] = ["default"]
                return DeterministicModel(**config_args)

            mock_get_class.return_value = compatible_model
            model = get_model("test-model", {"outputs": ["hello"]})
            assert isinstance(model, DeterministicModel)
            assert model.config.outputs == ["hello"]
            assert model.config.model_name == "test-model"

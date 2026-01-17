import os
from unittest.mock import patch

import pytest

from minisweagent.models import GlobalModelStats, get_model, get_model_class, get_model_name
from minisweagent.models.test_models import DeterministicModel


class TestGetModelName:
    # Common config used across tests - model_name should be direct, not nested under "model"
    CONFIG_WITH_MODEL_NAME = {"model_name": "config-model"}

    def test_input_model_name_takes_precedence(self):
        """Test that explicit input_model_name overrides all other sources."""
        with patch.dict(os.environ, {"MSWEA_MODEL_NAME": "env-model"}):
            assert get_model_name("input-model", self.CONFIG_WITH_MODEL_NAME) == "input-model"

    def test_config_takes_precedence_over_env(self):
        """Test that config takes precedence over environment variable."""
        with patch.dict(os.environ, {"MSWEA_MODEL_NAME": "env-model"}):
            assert get_model_name(None, self.CONFIG_WITH_MODEL_NAME) == "config-model"

    def test_env_var_fallback(self):
        """Test that environment variable is used when no config provided."""
        with patch.dict(os.environ, {"MSWEA_MODEL_NAME": "env-model"}):
            assert get_model_name(None, {}) == "env-model"

    def test_config_fallback(self):
        """Test that config model name is used when input and env are missing."""
        with patch.dict(os.environ, {}, clear=True):
            assert get_model_name(None, self.CONFIG_WITH_MODEL_NAME) == "config-model"

    def test_raises_error_when_no_model_configured(self):
        """Test that ValueError is raised when no model is configured anywhere."""
        with patch.dict(os.environ, {}, clear=True):
            with pytest.raises(
                ValueError, match="No default model set. Please run `mini-extra config setup` to set one."
            ):
                get_model_name(None, {})

            with pytest.raises(
                ValueError, match="No default model set. Please run `mini-extra config setup` to set one."
            ):
                get_model_name(None, None)


class TestGetModelClass:
    def test_anthropic_model_selection(self):
        """Test that anthropic-related model names return LitellmModel by default."""
        from minisweagent.models.litellm_model import LitellmModel

        for name in ["anthropic", "sonnet", "opus", "claude-sonnet", "claude-opus"]:
            assert get_model_class(name) == LitellmModel

    def test_litellm_model_fallback(self):
        """Test that non-anthropic model names return LitellmModel."""
        from minisweagent.models.litellm_model import LitellmModel

        for name in ["gpt-4", "gpt-3.5-turbo", "llama2", "random-model"]:
            assert get_model_class(name) == LitellmModel

    def test_partial_matches(self):
        """Test that partial string matches work correctly."""
        from minisweagent.models.litellm_model import LitellmModel

        assert get_model_class("my-anthropic-model") == LitellmModel
        assert get_model_class("sonnet-latest") == LitellmModel
        assert get_model_class("opus-v2") == LitellmModel
        assert get_model_class("gpt-anthropic-style") == LitellmModel
        assert get_model_class("totally-different") == LitellmModel

    def test_litellm_response_model_selection(self):
        """Test that litellm_response model class can be selected."""
        from minisweagent.models.litellm_response_api_model import LitellmResponseAPIModel

        assert get_model_class("any-model", "litellm_response") == LitellmResponseAPIModel


class TestGetModel:
    def test_config_deep_copy(self):
        """Test that get_model preserves original config via deep copy."""
        original_config = {"model_kwargs": {"api_key": "original"}, "outputs": ["test"]}

        with patch("minisweagent.models.get_model_class") as mock_get_class:
            mock_get_class.return_value = lambda **kwargs: DeterministicModel(outputs=["test"], model_name="test")
            get_model("test-model", original_config)
            assert original_config["model_kwargs"]["api_key"] == "original"
            assert "model_name" not in original_config

    def test_integration_with_compatible_model(self):
        """Test get_model works end-to-end with a model that handles extra kwargs."""
        with patch("minisweagent.models.get_model_class") as mock_get_class:

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

    def test_env_var_overrides_config_api_key(self):
        """Test that MSWEA_MODEL_API_KEY overrides config api_key."""
        with patch.dict(os.environ, {"MSWEA_MODEL_API_KEY": "env-key"}):
            config = {"model_kwargs": {"api_key": "config-key"}, "model_class": "litellm"}
            model = get_model("test-model", config)

            # LitellmModel stores the api_key in model_kwargs
            assert model.config.model_kwargs["api_key"] == "env-key"

    def test_config_api_key_used_when_no_env_var(self):
        """Test that config api_key is used when env var is not set."""
        with patch.dict(os.environ, {}, clear=True):
            config = {"model_kwargs": {"api_key": "config-key"}, "model_class": "litellm"}
            model = get_model("test-model", config)

            # LitellmModel stores the api_key in model_kwargs
            assert model.config.model_kwargs["api_key"] == "config-key"

    def test_env_var_sets_api_key_when_no_config_key(self):
        """Test that MSWEA_MODEL_API_KEY is used when config has no api_key."""
        with patch.dict(os.environ, {"MSWEA_MODEL_API_KEY": "env-key"}):
            config = {"model_class": "litellm"}
            model = get_model("test-model", config)

            # LitellmModel stores the api_key in model_kwargs
            assert model.config.model_kwargs["api_key"] == "env-key"

    def test_no_api_key_when_none_provided(self):
        """Test that no api_key is set when neither env var nor config provide one."""
        with patch.dict(os.environ, {}, clear=True):
            config = {"model_class": "litellm"}
            model = get_model("test-model", config)

            # LitellmModel should not have api_key when none provided
            model_kwargs = getattr(model.config, "model_kwargs", {})
            assert "api_key" not in model_kwargs

    def test_get_deterministic_model(self):
        """Test that get_model can instantiate DeterministicModel via model_class parameter."""
        config = {"outputs": ["hello", "world"], "cost_per_call": 2.0}
        model = get_model("test-model", config | {"model_class": "deterministic"})

        assert isinstance(model, DeterministicModel)
        assert model.config.outputs == ["hello", "world"]
        assert model.config.cost_per_call == 2.0
        assert model.config.model_name == "test-model"


class TestGlobalModelStats:
    def test_prints_cost_limit_when_set(self, capsys):
        """Test that cost limit is printed when MSWEA_GLOBAL_COST_LIMIT is set."""
        with patch.dict(os.environ, {"MSWEA_GLOBAL_COST_LIMIT": "5.5"}, clear=True):
            GlobalModelStats()
            captured = capsys.readouterr()
            assert "Global cost/call limit: $5.5000 / 0" in captured.out

    def test_prints_call_limit_when_set(self, capsys):
        """Test that call limit is printed when MSWEA_GLOBAL_CALL_LIMIT is set."""
        with patch.dict(os.environ, {"MSWEA_GLOBAL_CALL_LIMIT": "10"}, clear=True):
            GlobalModelStats()
            captured = capsys.readouterr()
            assert "Global cost/call limit: $0.0000 / 10" in captured.out

    def test_prints_both_limits_when_both_set(self, capsys):
        """Test that both limits are printed when both environment variables are set."""
        with patch.dict(os.environ, {"MSWEA_GLOBAL_COST_LIMIT": "2.5", "MSWEA_GLOBAL_CALL_LIMIT": "5"}, clear=True):
            GlobalModelStats()
            captured = capsys.readouterr()
            assert "Global cost/call limit: $2.5000 / 5" in captured.out

    def test_no_print_when_silent_startup_set(self, capsys):
        """Test that limits are not printed when MSWEA_SILENT_STARTUP is set."""
        with patch.dict(
            os.environ,
            {"MSWEA_GLOBAL_COST_LIMIT": "5.0", "MSWEA_GLOBAL_CALL_LIMIT": "10", "MSWEA_SILENT_STARTUP": "1"},
            clear=True,
        ):
            GlobalModelStats()
            captured = capsys.readouterr()
            assert "Global cost/call limit" not in captured.out

    def test_no_print_when_no_limits_set(self, capsys):
        """Test that nothing is printed when no limits are set."""
        with patch.dict(os.environ, {}, clear=True):
            GlobalModelStats()
            captured = capsys.readouterr()
            assert "Global cost/call limit" not in captured.out

    def test_no_print_when_limits_are_zero(self, capsys):
        """Test that nothing is printed when limits are explicitly set to zero."""
        with patch.dict(os.environ, {"MSWEA_GLOBAL_COST_LIMIT": "0", "MSWEA_GLOBAL_CALL_LIMIT": "0"}, clear=True):
            GlobalModelStats()
            captured = capsys.readouterr()
            assert "Global cost/call limit" not in captured.out

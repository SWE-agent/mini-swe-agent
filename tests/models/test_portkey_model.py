import os
from unittest.mock import MagicMock, patch

import pytest

from minisweagent.models import GLOBAL_MODEL_STATS
from minisweagent.models.portkey_model import PortkeyModel, PortkeyModelConfig


def test_portkey_model_missing_api_key():
    """Test that PortkeyModel raises ValueError when no API key is provided."""
    with patch("minisweagent.models.portkey_model.Portkey"):
        with patch.dict(os.environ, {}, clear=True):
            with pytest.raises(ValueError, match="Portkey API key is required"):
                PortkeyModel(model_name="gpt-4o")


def test_portkey_model_config():
    """Test PortkeyModelConfig creation."""
    config = PortkeyModelConfig(model_name="gpt-4o", model_kwargs={"temperature": 0.7})
    assert config.model_name == "gpt-4o"
    assert config.model_kwargs == {"temperature": 0.7}


def test_portkey_model_initialization():
    """Test PortkeyModel initialization with mocked Portkey."""
    mock_portkey_class = MagicMock()
    mock_client = MagicMock()
    mock_portkey_class.return_value = mock_client

    with patch("minisweagent.models.portkey_model.Portkey", mock_portkey_class):
        with patch.dict(os.environ, {"PORTKEY_API_KEY": "test-key", "PORTKEY_VIRTUAL_KEY": "test-virtual"}):
            model = PortkeyModel(model_name="gpt-4o")

            assert model.config.model_name == "gpt-4o"
            assert model.cost == 0.0
            assert model.n_calls == 0

            # Verify Portkey was called with correct parameters
            mock_portkey_class.assert_called_once_with(api_key="test-key", virtual_key="test-virtual")


def test_portkey_model_query():
    """Test PortkeyModel.query method with mocked response."""
    mock_portkey_class = MagicMock()
    mock_client = MagicMock()
    mock_response = MagicMock()
    mock_choice = MagicMock()
    mock_message = MagicMock()

    mock_message.content = "Hello! How can I help you?"
    mock_choice.message = mock_message
    mock_response.choices = [mock_choice]
    mock_response.model_dump.return_value = {"test": "response"}

    mock_client.chat.completions.create.return_value = mock_response
    mock_portkey_class.return_value = mock_client

    with patch("minisweagent.models.portkey_model.Portkey", mock_portkey_class):
        with patch.dict(os.environ, {"PORTKEY_API_KEY": "test-key"}):
            with patch("minisweagent.models.portkey_model.litellm.cost_calculator.completion_cost") as mock_cost:
                mock_cost.return_value = 0.01

                model = PortkeyModel(model_name="gpt-4o")

                messages = [{"role": "user", "content": "Hello!"}]
                result = model.query(messages)

                assert result["content"] == "Hello! How can I help you?"
                assert result["extra"]["response"] == {"test": "response"}
                assert result["extra"]["cost"] == 0.01
                assert model.n_calls == 1
                assert model.cost == 0.01

                # Verify the API was called correctly
                mock_client.chat.completions.create.assert_called_once_with(model="gpt-4o", messages=messages)
                # Verify cost calculation was called
                mock_cost.assert_called_once_with(mock_response.model_copy(), model=None)


def test_portkey_model_get_template_vars():
    """Test PortkeyModel.get_template_vars method."""
    mock_portkey_class = MagicMock()
    mock_client = MagicMock()
    mock_portkey_class.return_value = mock_client

    with patch("minisweagent.models.portkey_model.Portkey", mock_portkey_class):
        with patch.dict(os.environ, {"PORTKEY_API_KEY": "test-key"}):
            model = PortkeyModel(model_name="gpt-4o", model_kwargs={"temperature": 0.7})

            template_vars = model.get_template_vars()

            assert template_vars["model_name"] == "gpt-4o"
            assert template_vars["model_kwargs"] == {"temperature": 0.7}
            assert template_vars["n_model_calls"] == 0
            assert template_vars["model_cost"] == 0.0


def test_portkey_model_cost_tracking_ignore_errors():
    """Test that models work with cost_tracking='ignore_errors'."""
    mock_portkey_class = MagicMock()
    mock_client = MagicMock()
    mock_response = MagicMock()
    mock_choice = MagicMock()
    mock_message = MagicMock()

    mock_message.content = "Test response"
    mock_choice.message = mock_message
    mock_response.choices = [mock_choice]
    mock_response.model_dump.return_value = {"test": "response"}

    mock_client.chat.completions.create.return_value = mock_response
    mock_portkey_class.return_value = mock_client

    with patch("minisweagent.models.portkey_model.Portkey", mock_portkey_class):
        with patch.dict(os.environ, {"PORTKEY_API_KEY": "test-key"}):
            model = PortkeyModel(model_name="gpt-4o", cost_tracking="ignore_errors")

            initial_cost = GLOBAL_MODEL_STATS.cost

            with patch(
                "minisweagent.models.portkey_model.litellm.cost_calculator.completion_cost",
                side_effect=ValueError("Model not found"),
            ):
                messages = [{"role": "user", "content": "test"}]
                result = model.query(messages)

                assert result["content"] == "Test response"
                assert result["extra"]["cost"] == 0.0
                assert model.cost == 0.0
                assert model.n_calls == 1
                assert GLOBAL_MODEL_STATS.cost == initial_cost


def test_portkey_model_cost_validation_error():
    """Test that cost calculation errors raise RuntimeError when cost tracking is enabled."""
    mock_portkey_class = MagicMock()
    mock_client = MagicMock()
    mock_response = MagicMock()
    mock_choice = MagicMock()
    mock_message = MagicMock()
    mock_usage = MagicMock()

    mock_message.content = "Test response"
    mock_choice.message = mock_message
    mock_response.choices = [mock_choice]
    mock_response.model_dump.return_value = {"test": "response"}
    mock_response.model_copy.return_value = mock_response
    mock_response.usage = mock_usage
    mock_usage.prompt_tokens = 10
    mock_usage.completion_tokens = 20
    mock_usage.total_tokens = 30

    mock_client.chat.completions.create.return_value = mock_response
    mock_portkey_class.return_value = mock_client

    with patch("minisweagent.models.portkey_model.Portkey", mock_portkey_class):
        with patch.dict(os.environ, {"PORTKEY_API_KEY": "test-key"}):
            model = PortkeyModel(model_name="gpt-4o")

            with patch("minisweagent.models.portkey_model.litellm.cost_calculator.completion_cost") as mock_cost:
                mock_cost.side_effect = ValueError("Model not found")

                messages = [{"role": "user", "content": "test"}]

                with pytest.raises(RuntimeError) as exc_info:
                    model.query(messages)

                assert "Error calculating cost" in str(exc_info.value)
                assert "MSWEA_COST_TRACKING='ignore_errors'" in str(exc_info.value)

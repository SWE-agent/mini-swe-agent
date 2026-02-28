import json
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

import litellm
import pytest

from minisweagent.models import GLOBAL_MODEL_STATS
from minisweagent.models.litellm_model import LitellmModel
from minisweagent.models.litellm_response_api_model import LitellmResponseAPIModel
from minisweagent.models.utils.openai_utils import coerce_responses_text


def test_authentication_error_enhanced_message():
    """Test that AuthenticationError gets enhanced with config set instruction."""
    model = LitellmModel(model_name="gpt-4")

    # Create a mock exception that behaves like AuthenticationError
    original_error = Mock(spec=litellm.exceptions.AuthenticationError)
    original_error.message = "Invalid API key"

    with patch("litellm.completion") as mock_completion:
        # Make completion raise the mock error
        def side_effect(*args, **kwargs):
            raise litellm.exceptions.AuthenticationError("Invalid API key", llm_provider="openai", model="gpt-4")

        mock_completion.side_effect = side_effect

        with pytest.raises(litellm.exceptions.AuthenticationError) as exc_info:
            model._query([{"role": "user", "content": "test"}])

        # Check that the error message was enhanced
        assert "You can permanently set your API key with `mini-extra config set KEY VALUE`." in str(exc_info.value)


def test_model_registry_loading():
    """Test that custom model registry is loaded and registered when provided."""
    model_costs = {
        "my-custom-model": {
            "max_tokens": 4096,
            "input_cost_per_token": 0.0001,
            "output_cost_per_token": 0.0002,
            "litellm_provider": "openai",
            "mode": "chat",
        }
    }

    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(model_costs, f)
        registry_path = f.name

    try:
        with patch("litellm.utils.register_model") as mock_register:
            _model = LitellmModel(model_name="my-custom-model", litellm_model_registry=Path(registry_path))

            # Verify register_model was called with the correct data
            mock_register.assert_called_once_with(model_costs)
    except Exception as e:
        print(e)
        raise e
    finally:
        Path(registry_path).unlink()


def test_model_registry_none():
    """Test that no registry loading occurs when litellm_model_registry is None."""
    with patch("litellm.register_model") as mock_register:
        _model = LitellmModel(model_name="gpt-4", litellm_model_registry=None)

        # Verify register_model was not called
        mock_register.assert_not_called()


def test_model_registry_not_provided():
    """Test that no registry loading occurs when litellm_model_registry is not provided."""
    with patch("litellm.register_model") as mock_register:
        _model = LitellmModel(model_name="gpt-4o")

        # Verify register_model was not called
        mock_register.assert_not_called()


def test_litellm_model_cost_tracking_ignore_errors():
    """Test that models work with cost_tracking='ignore_errors'."""
    model = LitellmModel(model_name="gpt-4o", cost_tracking="ignore_errors")

    initial_cost = GLOBAL_MODEL_STATS.cost

    with patch("litellm.completion") as mock_completion:
        mock_response = Mock()
        mock_response.choices = [Mock(message=Mock(content="Test response"))]
        mock_response.model_dump.return_value = {"test": "response"}
        mock_completion.return_value = mock_response

        with patch("litellm.cost_calculator.completion_cost", side_effect=ValueError("Model not found")):
            messages = [{"role": "user", "content": "test"}]
            result = model.query(messages)

            assert result["content"] == "Test response"
            assert model.cost == 0.0
            assert model.n_calls == 1
            assert GLOBAL_MODEL_STATS.cost == initial_cost


def test_litellm_model_cost_validation_zero_cost():
    """Test that zero cost raises error when cost tracking is enabled."""
    model = LitellmModel(model_name="gpt-4o")

    with patch("litellm.completion") as mock_completion:
        mock_response = Mock()
        mock_response.choices = [Mock(message=Mock(content="Test response"))]
        mock_response.model_dump.return_value = {"test": "response"}
        mock_completion.return_value = mock_response

        with patch("litellm.cost_calculator.completion_cost", return_value=0.0):
            messages = [{"role": "user", "content": "test"}]

            with pytest.raises(RuntimeError) as exc_info:
                model.query(messages)

            assert "Cost must be > 0.0, got 0.0" in str(exc_info.value)
            assert "MSWEA_COST_TRACKING='ignore_errors'" in str(exc_info.value)


def test_response_api_model_basic_query():
    """Test that Response API model uses litellm.responses and tracks previous_response_id."""
    model = LitellmResponseAPIModel(model_name="gpt-5-mini")

    with (
        patch("litellm.responses") as mock_responses,
        patch("litellm.cost_calculator.completion_cost", return_value=0.01),
    ):
        from openai.types.responses.response_output_message import ResponseOutputMessage

        mock_response = Mock()
        mock_response.id = "resp_123"
        mock_output_message = Mock(spec=ResponseOutputMessage)
        mock_content = Mock()
        mock_content.text = "Test response"
        mock_output_message.content = [mock_content]
        mock_response.output = [mock_output_message]
        mock_response.output_text = None
        mock_responses.return_value = mock_response

        messages = [{"role": "user", "content": "test"}]
        result = model.query(messages)

        assert result["content"] == "Test response"
        assert model._previous_response_id == "resp_123"
        mock_responses.assert_called_once_with(model="gpt-5-mini", input=messages, previous_response_id=None)


def test_response_api_model_with_previous_id():
    """Test that Response API model passes previous_response_id on subsequent calls."""
    model = LitellmResponseAPIModel(model_name="gpt-5-mini")

    with (
        patch("litellm.responses") as mock_responses,
        patch("litellm.cost_calculator.completion_cost", return_value=0.01),
    ):
        from openai.types.responses.response_output_message import ResponseOutputMessage

        # First call
        mock_response1 = Mock()
        mock_response1.id = "resp_123"
        mock_output_message1 = Mock(spec=ResponseOutputMessage)
        mock_content1 = Mock()
        mock_content1.text = "First response"
        mock_output_message1.content = [mock_content1]
        mock_response1.output = [mock_output_message1]
        mock_response1.output_text = None
        mock_responses.return_value = mock_response1

        messages1 = [{"role": "user", "content": "first"}]
        model.query(messages1)

        # Second call
        mock_response2 = Mock()
        mock_response2.id = "resp_456"
        mock_output_message2 = Mock(spec=ResponseOutputMessage)
        mock_content2 = Mock()
        mock_content2.text = "Second response"
        mock_output_message2.content = [mock_content2]
        mock_response2.output = [mock_output_message2]
        mock_response2.output_text = None
        mock_responses.return_value = mock_response2

        messages2 = [
            {"role": "user", "content": "first"},
            {"role": "assistant", "content": "First response"},
            {"role": "user", "content": "second"},
        ]
        result = model.query(messages2)

        assert result["content"] == "Second response"
        assert model._previous_response_id == "resp_456"
        # On second call, should only pass the last message
        assert mock_responses.call_args[1]["input"] == [{"role": "user", "content": "second"}]
        assert mock_responses.call_args[1]["previous_response_id"] == "resp_123"


def test_response_api_model_output_text_field():
    """Test that Response API model uses output_text field when available."""
    model = LitellmResponseAPIModel(model_name="gpt-5-mini")

    with (
        patch("litellm.responses") as mock_responses,
        patch("litellm.cost_calculator.completion_cost", return_value=0.01),
    ):
        mock_response = Mock()
        mock_response.id = "resp_789"
        mock_response.output_text = "Direct output text"
        mock_responses.return_value = mock_response

        messages = [{"role": "user", "content": "test"}]
        result = model.query(messages)

        assert result["content"] == "Direct output text"


def test_response_api_model_multiple_output_messages():
    """Test that Response API model concatenates multiple output messages."""
    model = LitellmResponseAPIModel(model_name="gpt-5-mini")

    with (
        patch("litellm.responses") as mock_responses,
        patch("litellm.cost_calculator.completion_cost", return_value=0.01),
    ):
        mock_response = Mock()
        mock_response.id = "resp_999"
        mock_response.output_text = None

        # Create multiple output messages
        from openai.types.responses.response_output_message import ResponseOutputMessage

        mock_msg1 = Mock(spec=ResponseOutputMessage)
        mock_msg1.content = [Mock(text="First part")]
        mock_msg2 = Mock(spec=ResponseOutputMessage)
        mock_msg2.content = [Mock(text="Second part")]

        mock_response.output = [mock_msg1, mock_msg2]
        mock_responses.return_value = mock_response

        messages = [{"role": "user", "content": "test"}]
        result = model.query(messages)

        assert result["content"] == "First part\n\nSecond part"


def test_response_api_model_authentication_error():
    """Test that Response API model enhances AuthenticationError messages."""
    model = LitellmResponseAPIModel(model_name="gpt-5-mini")

    with patch("litellm.responses") as mock_responses:

        def side_effect(*args, **kwargs):
            raise litellm.exceptions.AuthenticationError("Invalid API key", llm_provider="openai", model="gpt-5-mini")

        mock_responses.side_effect = side_effect

        with pytest.raises(litellm.exceptions.AuthenticationError) as exc_info:
            model._query([{"role": "user", "content": "test"}])

        assert "You can permanently set your API key with `mini-extra config set KEY VALUE`." in str(exc_info.value)


def test_coerce_responses_text_with_output_text_field():
    """Test that coerce_responses_text uses output_text field when available."""
    mock_resp = Mock()
    mock_resp.output_text = "Direct output text"
    assert coerce_responses_text(mock_resp) == "Direct output text"


def test_coerce_responses_text_dict_single_content():
    """Test coerce_responses_text with dict format and single content item."""
    mock_resp = Mock()
    mock_resp.output_text = None
    mock_resp.output = [{"content": [{"text": "Test response"}]}]
    assert coerce_responses_text(mock_resp) == "Test response"


def test_coerce_responses_text_dict_multiple_content():
    """Test coerce_responses_text with dict format and multiple content items in one message."""
    mock_resp = Mock()
    mock_resp.output_text = None
    mock_resp.output = [{"content": [{"text": "First part"}, {"text": "Second part"}]}]
    assert coerce_responses_text(mock_resp) == "First part\n\nSecond part"


def test_coerce_responses_text_dict_multiple_messages():
    """Test coerce_responses_text with dict format and multiple messages."""
    mock_resp = Mock()
    mock_resp.output_text = None
    mock_resp.output = [{"content": [{"text": "Message 1"}]}, {"content": [{"text": "Message 2"}]}]
    assert coerce_responses_text(mock_resp) == "Message 1\n\nMessage 2"


def test_coerce_responses_text_object_format():
    """Test coerce_responses_text with ResponseOutputMessage objects."""
    from openai.types.responses.response_output_message import ResponseOutputMessage

    mock_resp = Mock()
    mock_resp.output_text = None
    mock_msg = Mock(spec=ResponseOutputMessage)
    mock_msg.content = [Mock(text="Object format response")]
    mock_resp.output = [mock_msg]
    assert coerce_responses_text(mock_resp) == "Object format response"


def test_coerce_responses_text_mixed_formats():
    """Test coerce_responses_text with both dict and object formats."""
    from openai.types.responses.response_output_message import ResponseOutputMessage

    mock_resp = Mock()
    mock_resp.output_text = None
    mock_msg = Mock(spec=ResponseOutputMessage)
    mock_msg.content = [Mock(text="Object response")]
    mock_resp.output = [{"content": [{"text": "Dict response"}]}, mock_msg]
    assert coerce_responses_text(mock_resp) == "Dict response\n\nObject response"


def test_coerce_responses_text_empty_response():
    """Test coerce_responses_text with empty output."""
    mock_resp = Mock()
    mock_resp.output_text = None
    mock_resp.output = []
    assert coerce_responses_text(mock_resp) == ""


def test_coerce_responses_text_no_text_fields():
    """Test coerce_responses_text when content items have no text."""
    mock_resp = Mock()
    mock_resp.output_text = None
    mock_resp.output = [{"content": [{"type": "image"}]}]
    assert coerce_responses_text(mock_resp) == ""


def test_coerce_responses_text_skip_non_dict_non_message():
    """Test coerce_responses_text skips items that are neither dict nor ResponseOutputMessage."""
    mock_resp = Mock()
    mock_resp.output_text = None
    mock_resp.output = ["invalid_item", {"content": [{"text": "Valid text"}]}, None]
    assert coerce_responses_text(mock_resp) == "Valid text"


def test_coerce_responses_text_empty_string_not_included():
    """Test coerce_responses_text skips empty text values."""
    mock_resp = Mock()
    mock_resp.output_text = None
    mock_resp.output = [{"content": [{"text": ""}, {"text": "Non-empty"}]}]
    assert coerce_responses_text(mock_resp) == "Non-empty"

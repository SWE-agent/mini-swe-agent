import os
from unittest.mock import MagicMock, Mock, patch

import pytest

from minisweagent.models import GLOBAL_MODEL_STATS
from minisweagent.models.portkey_response_api_model import PortkeyResponseAPIModel


def test_response_api_model_basic_query():
    """Test that Response API model uses client.responses and tracks previous_response_id."""
    mock_portkey_class = MagicMock()
    mock_client = MagicMock()
    mock_portkey_class.return_value = mock_client

    with (
        patch("minisweagent.models.portkey_model.Portkey", mock_portkey_class),
        patch.dict(os.environ, {"PORTKEY_API_KEY": "test-key"}),
        patch(
            "minisweagent.models.portkey_response_api_model.litellm.cost_calculator.completion_cost", return_value=0.01
        ),
    ):
        from openai.types.responses.response_output_message import ResponseOutputMessage

        mock_response = Mock()
        mock_response.id = "resp_123"
        mock_output_message = Mock(spec=ResponseOutputMessage)
        mock_content = Mock()
        # Response must include bash block to avoid FormatError from parse_action
        mock_content.text = "```mswea_bash_command\necho test\n```"
        mock_output_message.content = [mock_content]
        mock_response.output = [mock_output_message]
        mock_response.output_text = None
        mock_response.model_dump.return_value = {"id": "resp_123"}
        mock_client.responses.create.return_value = mock_response

        model = PortkeyResponseAPIModel(model_name="gpt-5-mini")
        messages = [{"role": "user", "content": "test"}]
        result = model.query(messages)

        assert result["content"] == "```mswea_bash_command\necho test\n```"
        assert result["extra"]["actions"] == ["echo test"]
        assert model._previous_response_id == "resp_123"
        mock_client.responses.create.assert_called_once_with(
            model="gpt-5-mini", input=messages, previous_response_id=None
        )


def test_response_api_model_with_previous_id():
    """Test that Response API model passes previous_response_id on subsequent calls."""
    mock_portkey_class = MagicMock()
    mock_client = MagicMock()
    mock_portkey_class.return_value = mock_client

    with (
        patch("minisweagent.models.portkey_model.Portkey", mock_portkey_class),
        patch.dict(os.environ, {"PORTKEY_API_KEY": "test-key"}),
        patch(
            "minisweagent.models.portkey_response_api_model.litellm.cost_calculator.completion_cost", return_value=0.01
        ),
    ):
        from openai.types.responses.response_output_message import ResponseOutputMessage

        # First call - response must include bash block
        mock_response1 = Mock()
        mock_response1.id = "resp_123"
        mock_output_message1 = Mock(spec=ResponseOutputMessage)
        mock_content1 = Mock()
        mock_content1.text = "```mswea_bash_command\necho first\n```"
        mock_output_message1.content = [mock_content1]
        mock_response1.output = [mock_output_message1]
        mock_response1.output_text = None
        mock_response1.model_dump.return_value = {"id": "resp_123"}
        mock_client.responses.create.return_value = mock_response1

        model = PortkeyResponseAPIModel(model_name="gpt-5-mini")
        messages1 = [{"role": "user", "content": "first"}]
        model.query(messages1)

        # Second call - response must include bash block
        mock_response2 = Mock()
        mock_response2.id = "resp_456"
        mock_output_message2 = Mock(spec=ResponseOutputMessage)
        mock_content2 = Mock()
        mock_content2.text = "```mswea_bash_command\necho second\n```"
        mock_output_message2.content = [mock_content2]
        mock_response2.output = [mock_output_message2]
        mock_response2.output_text = None
        mock_response2.model_dump.return_value = {"id": "resp_456"}
        mock_client.responses.create.return_value = mock_response2

        messages2 = [
            {"role": "user", "content": "first"},
            {"role": "assistant", "content": "```mswea_bash_command\necho first\n```"},
            {"role": "user", "content": "second"},
        ]
        result = model.query(messages2)

        assert result["content"] == "```mswea_bash_command\necho second\n```"
        assert model._previous_response_id == "resp_456"
        # On second call, should only pass the last message
        assert mock_client.responses.create.call_args[1]["input"] == [{"role": "user", "content": "second"}]
        assert mock_client.responses.create.call_args[1]["previous_response_id"] == "resp_123"


def test_response_api_model_output_text_field():
    """Test that Response API model uses output_text field when available."""
    mock_portkey_class = MagicMock()
    mock_client = MagicMock()
    mock_portkey_class.return_value = mock_client

    with (
        patch("minisweagent.models.portkey_model.Portkey", mock_portkey_class),
        patch.dict(os.environ, {"PORTKEY_API_KEY": "test-key"}),
        patch(
            "minisweagent.models.portkey_response_api_model.litellm.cost_calculator.completion_cost", return_value=0.01
        ),
    ):
        mock_response = Mock()
        mock_response.id = "resp_789"
        # Response must include bash block to avoid FormatError from parse_action
        mock_response.output_text = "```mswea_bash_command\necho direct\n```"
        mock_response.model_dump.return_value = {"id": "resp_789"}
        mock_client.responses.create.return_value = mock_response

        model = PortkeyResponseAPIModel(model_name="gpt-5-mini")
        messages = [{"role": "user", "content": "test"}]
        result = model.query(messages)

        assert result["content"] == "```mswea_bash_command\necho direct\n```"
        assert result["extra"]["actions"] == ["echo direct"]


def test_response_api_model_multiple_output_messages():
    """Test that Response API model concatenates multiple output messages."""
    mock_portkey_class = MagicMock()
    mock_client = MagicMock()
    mock_portkey_class.return_value = mock_client

    with (
        patch("minisweagent.models.portkey_model.Portkey", mock_portkey_class),
        patch.dict(os.environ, {"PORTKEY_API_KEY": "test-key"}),
        patch(
            "minisweagent.models.portkey_response_api_model.litellm.cost_calculator.completion_cost", return_value=0.01
        ),
    ):
        from openai.types.responses.response_output_message import ResponseOutputMessage

        mock_response = Mock()
        mock_response.id = "resp_999"
        mock_response.output_text = None
        mock_response.model_dump.return_value = {"id": "resp_999"}

        # Create multiple output messages - together they form a valid bash block
        mock_msg1 = Mock(spec=ResponseOutputMessage)
        mock_msg1.content = [Mock(text="First part\n```mswea_bash_command")]
        mock_msg2 = Mock(spec=ResponseOutputMessage)
        mock_msg2.content = [Mock(text="echo test\n```")]

        mock_response.output = [mock_msg1, mock_msg2]
        mock_client.responses.create.return_value = mock_response

        model = PortkeyResponseAPIModel(model_name="gpt-5-mini")
        messages = [{"role": "user", "content": "test"}]
        result = model.query(messages)

        assert result["content"] == "First part\n```mswea_bash_command\n\necho test\n```"
        assert result["extra"]["actions"] == ["echo test"]


def test_response_api_model_cost_tracking():
    """Test that Response API model tracks costs correctly."""
    mock_portkey_class = MagicMock()
    mock_client = MagicMock()
    mock_portkey_class.return_value = mock_client

    with (
        patch("minisweagent.models.portkey_model.Portkey", mock_portkey_class),
        patch.dict(os.environ, {"PORTKEY_API_KEY": "test-key"}),
        patch(
            "minisweagent.models.portkey_response_api_model.litellm.cost_calculator.completion_cost", return_value=0.05
        ),
    ):
        mock_response = Mock()
        mock_response.id = "resp_cost"
        # Response must include bash block to avoid FormatError from parse_action
        mock_response.output_text = "```mswea_bash_command\necho cost\n```"
        mock_response.model_dump.return_value = {"id": "resp_cost"}
        mock_client.responses.create.return_value = mock_response

        initial_global_cost = GLOBAL_MODEL_STATS.cost
        model = PortkeyResponseAPIModel(model_name="gpt-5-mini")

        messages = [{"role": "user", "content": "test"}]
        result = model.query(messages)

        assert result["extra"]["cost"] == 0.05
        assert model.cost == 0.05
        assert model.n_calls == 1
        assert GLOBAL_MODEL_STATS.cost == initial_global_cost + 0.05


def test_response_api_model_zero_cost_assertion():
    """Test that Response API model raises RuntimeError for zero cost."""
    mock_portkey_class = MagicMock()
    mock_client = MagicMock()
    mock_portkey_class.return_value = mock_client

    with (
        patch("minisweagent.models.portkey_model.Portkey", mock_portkey_class),
        patch.dict(os.environ, {"PORTKEY_API_KEY": "test-key"}),
        patch(
            "minisweagent.models.portkey_response_api_model.litellm.cost_calculator.completion_cost", return_value=0.0
        ),
    ):
        mock_response = Mock()
        mock_response.id = "resp_zero"
        mock_response.output_text = "Response"
        mock_response.model_dump.return_value = {"id": "resp_zero"}
        mock_client.responses.create.return_value = mock_response

        model = PortkeyResponseAPIModel(model_name="gpt-5-mini")
        messages = [{"role": "user", "content": "test"}]

        with pytest.raises(RuntimeError, match="Error calculating cost"):
            model.query(messages)


def test_response_api_model_cache_control():
    """Test that Response API model applies cache control when configured."""
    mock_portkey_class = MagicMock()
    mock_client = MagicMock()
    mock_portkey_class.return_value = mock_client

    with (
        patch("minisweagent.models.portkey_model.Portkey", mock_portkey_class),
        patch.dict(os.environ, {"PORTKEY_API_KEY": "test-key"}),
        patch(
            "minisweagent.models.portkey_response_api_model.litellm.cost_calculator.completion_cost", return_value=0.01
        ),
        patch("minisweagent.models.portkey_response_api_model.set_cache_control") as mock_cache,
    ):
        mock_response = Mock()
        mock_response.id = "resp_cache"
        # Response must include bash block to avoid FormatError from parse_action
        mock_response.output_text = "```mswea_bash_command\necho cache\n```"
        mock_response.model_dump.return_value = {"id": "resp_cache"}
        mock_client.responses.create.return_value = mock_response

        messages_original = [{"role": "user", "content": "test"}]
        messages_cached = [{"role": "user", "content": "test", "cache_control": {"type": "ephemeral"}}]
        mock_cache.return_value = messages_cached

        model = PortkeyResponseAPIModel(model_name="gpt-5-mini", set_cache_control="default_end")
        model.query(messages_original)

        mock_cache.assert_called_once_with(messages_original, mode="default_end")


def test_response_api_model_with_model_kwargs():
    """Test that Response API model passes model_kwargs to the API."""
    mock_portkey_class = MagicMock()
    mock_client = MagicMock()
    mock_portkey_class.return_value = mock_client

    with (
        patch("minisweagent.models.portkey_model.Portkey", mock_portkey_class),
        patch.dict(os.environ, {"PORTKEY_API_KEY": "test-key"}),
        patch(
            "minisweagent.models.portkey_response_api_model.litellm.cost_calculator.completion_cost", return_value=0.01
        ),
    ):
        mock_response = Mock()
        mock_response.id = "resp_kwargs"
        # Response must include bash block to avoid FormatError from parse_action
        mock_response.output_text = "```mswea_bash_command\necho kwargs\n```"
        mock_response.model_dump.return_value = {"id": "resp_kwargs"}
        mock_client.responses.create.return_value = mock_response

        model = PortkeyResponseAPIModel(model_name="gpt-5-mini", model_kwargs={"temperature": 0.7, "max_tokens": 100})
        messages = [{"role": "user", "content": "test"}]
        model.query(messages)

        call_kwargs = mock_client.responses.create.call_args[1]
        assert call_kwargs["temperature"] == 0.7
        assert call_kwargs["max_tokens"] == 100


def test_response_api_model_retry_on_rate_limit():
    """Test that Response API model retries on rate limit errors."""
    mock_portkey_class = MagicMock()
    mock_client = MagicMock()
    mock_portkey_class.return_value = mock_client

    with (
        patch("minisweagent.models.portkey_model.Portkey", mock_portkey_class),
        patch.dict(os.environ, {"PORTKEY_API_KEY": "test-key", "MSWEA_MODEL_RETRY_STOP_AFTER_ATTEMPT": "2"}),
        patch(
            "minisweagent.models.portkey_response_api_model.litellm.cost_calculator.completion_cost", return_value=0.01
        ),
    ):
        from openai.types.responses.response_output_message import ResponseOutputMessage

        call_count = 0

        def side_effect(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise Exception("Rate limit exceeded")
            mock_response = Mock()
            mock_response.id = "resp_retry"
            mock_output_message = Mock(spec=ResponseOutputMessage)
            mock_content = Mock()
            # Response must include bash block to avoid FormatError from parse_action
            mock_content.text = "```mswea_bash_command\necho 'Success after retry'\n```"
            mock_output_message.content = [mock_content]
            mock_response.output = [mock_output_message]
            mock_response.output_text = None
            mock_response.model_dump.return_value = {"id": "resp_retry"}
            return mock_response

        mock_client.responses.create.side_effect = side_effect

        model = PortkeyResponseAPIModel(model_name="gpt-5-mini")
        messages = [{"role": "user", "content": "test"}]
        result = model.query(messages)

        assert result["content"] == "```mswea_bash_command\necho 'Success after retry'\n```"
        assert call_count == 2


def test_response_api_model_no_retry_on_type_error():
    """Test that Response API model does not retry on TypeError."""
    mock_portkey_class = MagicMock()
    mock_client = MagicMock()
    mock_portkey_class.return_value = mock_client

    with (
        patch("minisweagent.models.portkey_model.Portkey", mock_portkey_class),
        patch.dict(os.environ, {"PORTKEY_API_KEY": "test-key"}),
    ):
        mock_client.responses.create.side_effect = TypeError("Invalid type")

        model = PortkeyResponseAPIModel(model_name="gpt-5-mini")
        messages = [{"role": "user", "content": "test"}]

        with pytest.raises(TypeError, match="Invalid type"):
            model.query(messages)

        # Should only be called once (no retries)
        assert mock_client.responses.create.call_count == 1

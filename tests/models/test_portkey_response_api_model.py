import os
from unittest.mock import MagicMock, Mock, patch

import pytest

from minisweagent.models import GLOBAL_MODEL_STATS
from minisweagent.models.portkey_response_api_model import PortkeyResponseAPIModel
from minisweagent.models.utils.actions_toolcall_response import BASH_TOOL_RESPONSE_API


def test_response_api_model_basic_query():
    """Test that Response API model uses client.responses with stateless interface."""
    mock_portkey_class = MagicMock()
    mock_client = MagicMock()
    mock_portkey_class.return_value = mock_client

    with (
        patch("minisweagent.models.portkey_response_api_model.Portkey", mock_portkey_class),
        patch.dict(os.environ, {"PORTKEY_API_KEY": "test-key"}),
        patch(
            "minisweagent.models.portkey_response_api_model.litellm.cost_calculator.completion_cost", return_value=0.01
        ),
    ):
        mock_response = Mock()
        mock_response.id = "resp_123"
        mock_response.output = [
            {"type": "function_call", "call_id": "call_abc", "name": "bash", "arguments": '{"command": "echo test"}'}
        ]
        mock_response.model_dump.return_value = {
            "id": "resp_123",
            "output": mock_response.output,
        }
        mock_client.responses.create.return_value = mock_response

        model = PortkeyResponseAPIModel(model_name="gpt-5-mini")
        messages = [{"role": "user", "content": "test"}]
        result = model.query(messages)

        assert result["extra"]["actions"] == [{"command": "echo test", "tool_call_id": "call_abc"}]
        mock_client.responses.create.assert_called_once_with(
            model="gpt-5-mini", input=messages, tools=[BASH_TOOL_RESPONSE_API]
        )


def test_response_api_model_stateless_flattens_response():
    """Test that Response API model flattens response objects for stateless API."""
    mock_portkey_class = MagicMock()
    mock_client = MagicMock()
    mock_portkey_class.return_value = mock_client

    with (
        patch("minisweagent.models.portkey_response_api_model.Portkey", mock_portkey_class),
        patch.dict(os.environ, {"PORTKEY_API_KEY": "test-key"}),
        patch(
            "minisweagent.models.portkey_response_api_model.litellm.cost_calculator.completion_cost", return_value=0.01
        ),
    ):
        mock_response = Mock()
        mock_response.id = "resp_456"
        mock_response.output = [
            {"type": "function_call", "call_id": "call_2", "name": "bash", "arguments": '{"command": "echo second"}'}
        ]
        mock_response.model_dump.return_value = {"id": "resp_456", "output": mock_response.output}
        mock_client.responses.create.return_value = mock_response

        model = PortkeyResponseAPIModel(model_name="gpt-5-mini")
        messages = [
            {"type": "message", "role": "user", "content": [{"type": "input_text", "text": "first"}]},
            {
                "object": "response",
                "output": [
                    {
                        "type": "function_call",
                        "call_id": "call_1",
                        "name": "bash",
                        "arguments": '{"command": "echo first"}',
                    },
                ],
                "extra": {"actions": [{"command": "echo first", "tool_call_id": "call_1"}]},
            },
            {"type": "function_call_output", "call_id": "call_1", "output": "first"},
            {"type": "message", "role": "user", "content": [{"type": "input_text", "text": "second"}]},
        ]
        result = model.query(messages)

        assert result["extra"]["actions"] == [{"command": "echo second", "tool_call_id": "call_2"}]
        # Verify that response objects are flattened and extra is stripped
        expected_input = [
            {"type": "message", "role": "user", "content": [{"type": "input_text", "text": "first"}]},
            {"type": "function_call", "call_id": "call_1", "name": "bash", "arguments": '{"command": "echo first"}'},
            {"type": "function_call_output", "call_id": "call_1", "output": "first"},
            {"type": "message", "role": "user", "content": [{"type": "input_text", "text": "second"}]},
        ]
        assert mock_client.responses.create.call_args[1]["input"] == expected_input


def test_response_api_model_multiple_tool_calls():
    """Test that Response API model handles multiple tool calls."""
    mock_portkey_class = MagicMock()
    mock_client = MagicMock()
    mock_portkey_class.return_value = mock_client

    with (
        patch("minisweagent.models.portkey_response_api_model.Portkey", mock_portkey_class),
        patch.dict(os.environ, {"PORTKEY_API_KEY": "test-key"}),
        patch(
            "minisweagent.models.portkey_response_api_model.litellm.cost_calculator.completion_cost", return_value=0.01
        ),
    ):
        mock_response = Mock()
        mock_response.id = "resp_789"
        mock_response.output = [
            {"type": "function_call", "call_id": "call_1", "name": "bash", "arguments": '{"command": "echo first"}'},
            {"type": "function_call", "call_id": "call_2", "name": "bash", "arguments": '{"command": "echo second"}'},
        ]
        mock_response.model_dump.return_value = {
            "id": "resp_789",
            "output": mock_response.output,
        }
        mock_client.responses.create.return_value = mock_response

        model = PortkeyResponseAPIModel(model_name="gpt-5-mini")
        messages = [{"role": "user", "content": "test"}]
        result = model.query(messages)

        assert result["extra"]["actions"] == [
            {"command": "echo first", "tool_call_id": "call_1"},
            {"command": "echo second", "tool_call_id": "call_2"},
        ]


def test_response_api_model_cost_tracking():
    """Test that Response API model tracks costs correctly."""
    mock_portkey_class = MagicMock()
    mock_client = MagicMock()
    mock_portkey_class.return_value = mock_client

    with (
        patch("minisweagent.models.portkey_response_api_model.Portkey", mock_portkey_class),
        patch.dict(os.environ, {"PORTKEY_API_KEY": "test-key"}),
        patch(
            "minisweagent.models.portkey_response_api_model.litellm.cost_calculator.completion_cost", return_value=0.05
        ),
    ):
        mock_response = Mock()
        mock_response.id = "resp_cost"
        mock_response.output = [
            {"type": "function_call", "call_id": "call_cost", "name": "bash", "arguments": '{"command": "echo cost"}'}
        ]
        mock_response.model_dump.return_value = {"id": "resp_cost", "output": mock_response.output}
        mock_client.responses.create.return_value = mock_response

        initial_global_cost = GLOBAL_MODEL_STATS.cost
        model = PortkeyResponseAPIModel(model_name="gpt-5-mini")

        messages = [{"role": "user", "content": "test"}]
        result = model.query(messages)

        assert result["extra"]["cost"] == 0.05
        assert GLOBAL_MODEL_STATS.cost == initial_global_cost + 0.05


def test_response_api_model_zero_cost_assertion():
    """Test that Response API model raises RuntimeError for zero cost."""
    mock_portkey_class = MagicMock()
    mock_client = MagicMock()
    mock_portkey_class.return_value = mock_client

    with (
        patch("minisweagent.models.portkey_response_api_model.Portkey", mock_portkey_class),
        patch.dict(os.environ, {"PORTKEY_API_KEY": "test-key"}),
        patch(
            "minisweagent.models.portkey_response_api_model.litellm.cost_calculator.completion_cost", return_value=0.0
        ),
    ):
        mock_response = Mock()
        mock_response.id = "resp_zero"
        mock_response.output = [
            {"type": "function_call", "call_id": "call_zero", "name": "bash", "arguments": '{"command": "echo test"}'}
        ]
        mock_response.model_dump.return_value = {"id": "resp_zero", "output": mock_response.output}
        mock_client.responses.create.return_value = mock_response

        model = PortkeyResponseAPIModel(model_name="gpt-5-mini")
        messages = [{"role": "user", "content": "test"}]

        with pytest.raises(RuntimeError, match="Error calculating cost"):
            model.query(messages)


def test_response_api_model_with_model_kwargs():
    """Test that Response API model passes model_kwargs to the API."""
    mock_portkey_class = MagicMock()
    mock_client = MagicMock()
    mock_portkey_class.return_value = mock_client

    with (
        patch("minisweagent.models.portkey_response_api_model.Portkey", mock_portkey_class),
        patch.dict(os.environ, {"PORTKEY_API_KEY": "test-key"}),
        patch(
            "minisweagent.models.portkey_response_api_model.litellm.cost_calculator.completion_cost", return_value=0.01
        ),
    ):
        mock_response = Mock()
        mock_response.id = "resp_kwargs"
        mock_response.output = [
            {"type": "function_call", "call_id": "call_kw", "name": "bash", "arguments": '{"command": "echo kwargs"}'}
        ]
        mock_response.model_dump.return_value = {"id": "resp_kwargs", "output": mock_response.output}
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
        patch("minisweagent.models.portkey_response_api_model.Portkey", mock_portkey_class),
        patch.dict(os.environ, {"PORTKEY_API_KEY": "test-key", "MSWEA_MODEL_RETRY_STOP_AFTER_ATTEMPT": "2"}),
        patch(
            "minisweagent.models.portkey_response_api_model.litellm.cost_calculator.completion_cost", return_value=0.01
        ),
    ):
        call_count = 0

        def side_effect(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise Exception("Rate limit exceeded")
            mock_response = Mock()
            mock_response.id = "resp_retry"
            mock_response.output = [
                {
                    "type": "function_call",
                    "call_id": "call_retry",
                    "name": "bash",
                    "arguments": '{"command": "echo Success after retry"}',
                }
            ]
            mock_response.model_dump.return_value = {
                "id": "resp_retry",
                "output": mock_response.output,
            }
            return mock_response

        mock_client.responses.create.side_effect = side_effect

        model = PortkeyResponseAPIModel(model_name="gpt-5-mini")
        messages = [{"role": "user", "content": "test"}]
        result = model.query(messages)

        assert result["extra"]["actions"] == [{"command": "echo Success after retry", "tool_call_id": "call_retry"}]
        assert call_count == 2


def test_response_api_model_no_retry_on_type_error():
    """Test that Response API model does not retry on TypeError."""
    mock_portkey_class = MagicMock()
    mock_client = MagicMock()
    mock_portkey_class.return_value = mock_client

    with (
        patch("minisweagent.models.portkey_response_api_model.Portkey", mock_portkey_class),
        patch.dict(os.environ, {"PORTKEY_API_KEY": "test-key"}),
    ):
        mock_client.responses.create.side_effect = TypeError("Invalid type")

        model = PortkeyResponseAPIModel(model_name="gpt-5-mini")
        messages = [{"role": "user", "content": "test"}]

        with pytest.raises(TypeError, match="Invalid type"):
            model.query(messages)

        # Should only be called once (no retries)
        assert mock_client.responses.create.call_count == 1


def test_response_api_model_serialize():
    """Test that Response API model serializes correctly."""
    mock_portkey_class = MagicMock()
    mock_client = MagicMock()
    mock_portkey_class.return_value = mock_client

    with (
        patch("minisweagent.models.portkey_response_api_model.Portkey", mock_portkey_class),
        patch.dict(os.environ, {"PORTKEY_API_KEY": "test-key"}),
    ):
        model = PortkeyResponseAPIModel(model_name="gpt-5-mini")
        serialized = model.serialize()

        assert serialized["info"]["config"]["model"]["model_name"] == "gpt-5-mini"
        assert "PortkeyResponseAPIModel" in serialized["info"]["config"]["model_type"]


def test_response_api_model_get_template_vars():
    """Test that Response API model returns template vars from config."""
    mock_portkey_class = MagicMock()
    mock_client = MagicMock()
    mock_portkey_class.return_value = mock_client

    with (
        patch("minisweagent.models.portkey_response_api_model.Portkey", mock_portkey_class),
        patch.dict(os.environ, {"PORTKEY_API_KEY": "test-key"}),
    ):
        model = PortkeyResponseAPIModel(model_name="gpt-5-mini")
        template_vars = model.get_template_vars()

        assert template_vars["model_name"] == "gpt-5-mini"

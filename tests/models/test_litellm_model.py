from unittest.mock import MagicMock, patch

import pytest

from minisweagent.exceptions import FormatError
from minisweagent.models.litellm_model import LitellmModel, LitellmModelConfig
from minisweagent.models.utils.actions_toolcall import BASH_TOOL


class TestLitellmModelConfig:
    def test_default_format_error_template(self):
        assert LitellmModelConfig(model_name="test").format_error_template == "{{ error }}"


def _mock_litellm_response(tool_calls):
    mock_response = MagicMock()
    mock_response.choices = [MagicMock()]
    mock_response.choices[0].message.tool_calls = tool_calls
    mock_response.choices[0].message.model_dump.return_value = {"role": "assistant", "content": None}
    mock_response.model_dump.return_value = {}
    return mock_response


class TestLitellmModel:
    @patch("minisweagent.models.litellm_model.litellm.completion")
    @patch("minisweagent.models.litellm_model.litellm.cost_calculator.completion_cost")
    def test_query_includes_bash_tool(self, mock_cost, mock_completion):
        tool_call = MagicMock()
        tool_call.function.name = "bash"
        tool_call.function.arguments = '{"command": "echo test"}'
        tool_call.id = "call_1"
        mock_completion.return_value = _mock_litellm_response([tool_call])
        mock_cost.return_value = 0.001

        model = LitellmModel(model_name="gpt-4")
        model.query([{"role": "user", "content": "test"}])

        mock_completion.assert_called_once()
        assert mock_completion.call_args.kwargs["tools"] == [BASH_TOOL]

    @patch("minisweagent.models.litellm_model.litellm.completion")
    @patch("minisweagent.models.litellm_model.litellm.cost_calculator.completion_cost")
    def test_parse_actions_valid_tool_call(self, mock_cost, mock_completion):
        tool_call = MagicMock()
        tool_call.function.name = "bash"
        tool_call.function.arguments = '{"command": "ls -la"}'
        tool_call.id = "call_abc"
        mock_completion.return_value = _mock_litellm_response([tool_call])
        mock_cost.return_value = 0.001

        model = LitellmModel(model_name="gpt-4")
        result = model.query([{"role": "user", "content": "list files"}])
        assert result["extra"]["actions"] == [{"command": "ls -la", "tool_call_id": "call_abc"}]

    @patch("minisweagent.models.litellm_model.litellm.completion")
    @patch("minisweagent.models.litellm_model.litellm.cost_calculator.completion_cost")
    def test_parse_actions_no_tool_calls_raises(self, mock_cost, mock_completion):
        mock_completion.return_value = _mock_litellm_response(None)
        mock_cost.return_value = 0.001

        model = LitellmModel(model_name="gpt-4")
        with pytest.raises(FormatError):
            model.query([{"role": "user", "content": "test"}])

    @patch("minisweagent.models.litellm_model.litellm.completion")
    @patch("minisweagent.models.litellm_model.litellm.cost_calculator.completion_cost")
    def test_query_merges_tool_calls_across_multiple_choices(self, mock_cost, mock_completion):
        text_choice = MagicMock()
        text_choice.message.model_dump.return_value = {
            "role": "assistant",
            "content": "Let me start by analyzing the codebase to find the relevant files.",
            "tool_calls": None,
        }

        tool_choice = MagicMock()
        tool_choice.message.model_dump.return_value = {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "call_multi",
                    "function": {"name": "bash", "arguments": '{"command":"find . -name \\"TargetFile.java\\""}'},
                }
            ],
        }

        mock_response = MagicMock()
        mock_response.choices = [text_choice, tool_choice]
        mock_response.model_dump.return_value = {}
        mock_completion.return_value = mock_response
        mock_cost.return_value = 0.001

        model = LitellmModel(model_name="github_copilot/claude-sonnet-4.6")
        result = model.query([{"role": "user", "content": "test"}])

        assert result["content"] == "Let me start by analyzing the codebase to find the relevant files."
        assert result["extra"]["actions"] == [{"command": 'find . -name "TargetFile.java"', "tool_call_id": "call_multi"}]

    def test_parse_actions_accepts_dict_tool_calls(self):
        model = LitellmModel(model_name="github_copilot/claude-sonnet-4.6")
        response = MagicMock()
        response.choices = [
            MagicMock(
                message=MagicMock(
                    model_dump=MagicMock(
                        return_value={
                            "role": "assistant",
                            "content": None,
                            "tool_calls": [
                                {
                                    "id": "call_dict",
                                    "function": {"name": "bash", "arguments": '{"command":"pwd"}'},
                                }
                            ],
                        }
                    )
                )
            )
        ]

        assert model._parse_actions(response) == [{"command": "pwd", "tool_call_id": "call_dict"}]

    def test_format_observation_messages(self):
        model = LitellmModel(model_name="gpt-4", observation_template="{{ output.output }}")
        message = {"extra": {"actions": [{"command": "echo test", "tool_call_id": "call_1"}]}}
        outputs = [{"output": "test output", "returncode": 0}]
        result = model.format_observation_messages(message, outputs)
        assert len(result) == 1
        assert result[0]["role"] == "tool"
        assert result[0]["tool_call_id"] == "call_1"
        assert result[0]["content"] == "test output"

    def test_format_observation_messages_no_actions(self):
        model = LitellmModel(model_name="gpt-4")
        result = model.format_observation_messages({"extra": {}}, [])
        assert result == []

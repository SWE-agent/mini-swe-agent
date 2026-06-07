from unittest.mock import MagicMock

import pytest

from minisweagent.exceptions import FormatError
from minisweagent.models.utils.actions_toolcall import (
    BASH_TOOL,
    format_toolcall_observation_messages,
    parse_toolcall_actions,
)


class TestParseToolcallActions:
    def test_empty_tool_calls_raises_format_error(self):
        with pytest.raises(FormatError) as exc_info:
            parse_toolcall_actions([], format_error_template="{{ error }}")
        assert "No tool calls found" in exc_info.value.messages[0]["content"]

    def test_none_tool_calls_raises_format_error(self):
        with pytest.raises(FormatError) as exc_info:
            parse_toolcall_actions(None, format_error_template="{{ error }}")
        assert "No tool calls found" in exc_info.value.messages[0]["content"]

    def test_valid_bash_tool_call(self):
        tool_call = MagicMock()
        tool_call.function.name = "bash"
        tool_call.function.arguments = '{"command": "echo hello"}'
        tool_call.id = "call_123"
        assert parse_toolcall_actions([tool_call], format_error_template="{{ error }}") == [
            {"command": "echo hello", "tool_call_id": "call_123"}
        ]

    def test_multiple_valid_tool_calls(self):
        calls = []
        for i in range(3):
            tc = MagicMock()
            tc.function.name = "bash"
            tc.function.arguments = f'{{"command": "cmd{i}"}}'
            tc.id = f"call_{i}"
            calls.append(tc)
        result = parse_toolcall_actions(calls, format_error_template="{{ error }}")
        assert len(result) == 3
        assert result[0] == {"command": "cmd0", "tool_call_id": "call_0"}
        assert result[2] == {"command": "cmd2", "tool_call_id": "call_2"}

    def test_unknown_tool_raises_format_error(self):
        tool_call = MagicMock()
        tool_call.function.name = "unknown_tool"
        tool_call.function.arguments = '{"command": "test"}'
        tool_call.id = "call_1"
        with pytest.raises(FormatError) as exc_info:
            parse_toolcall_actions([tool_call], format_error_template="{{ error }}")
        assert "Unknown tool 'unknown_tool'" in exc_info.value.messages[0]["content"]

    def test_invalid_json_raises_format_error(self):
        tool_call = MagicMock()
        tool_call.function.name = "bash"
        tool_call.function.arguments = "not valid json"
        tool_call.id = "call_1"
        with pytest.raises(FormatError) as exc_info:
            parse_toolcall_actions([tool_call], format_error_template="{{ error }}")
        assert "Error parsing tool call arguments" in exc_info.value.messages[0]["content"]

    def test_missing_command_raises_format_error(self):
        tool_call = MagicMock()
        tool_call.function.name = "bash"
        tool_call.function.arguments = '{"other_arg": "value"}'
        tool_call.id = "call_1"
        with pytest.raises(FormatError) as exc_info:
            parse_toolcall_actions([tool_call], format_error_template="{{ error }}")
        assert "Missing 'command' argument" in exc_info.value.messages[0]["content"]


class TestFormatToolcallObservationMessages:
    def test_basic_formatting(self):
        actions = [{"command": "echo test", "tool_call_id": "call_1"}]
        outputs = [{"output": "test output", "returncode": 0}]
        result = format_toolcall_observation_messages(
            actions=actions, outputs=outputs, observation_template="{{ output.output }}"
        )
        assert len(result) == 1
        assert result[0]["role"] == "tool"
        assert result[0]["tool_call_id"] == "call_1"
        assert result[0]["content"] == "test output"
        assert result[0]["extra"]["returncode"] == 0

    def test_multiple_outputs(self):
        actions = [
            {"command": "cmd1", "tool_call_id": "call_1"},
            {"command": "cmd2", "tool_call_id": "call_2"},
        ]
        outputs = [{"output": "out1", "returncode": 0}, {"output": "out2", "returncode": 1}]
        result = format_toolcall_observation_messages(
            actions=actions, outputs=outputs, observation_template="{{ output.output }}"
        )
        assert len(result) == 2
        assert result[0]["tool_call_id"] == "call_1"
        assert result[0]["content"] == "out1"
        assert result[1]["tool_call_id"] == "call_2"
        assert result[1]["content"] == "out2"

    def test_with_template_vars(self):
        actions = [{"command": "test", "tool_call_id": "call_1"}]
        outputs = [{"output": "result", "returncode": 0}]
        result = format_toolcall_observation_messages(
            actions=actions,
            outputs=outputs,
            observation_template="{{ output.output }} - {{ custom_var }}",
            template_vars={"custom_var": "extra_info"},
        )
        assert result[0]["content"] == "result - extra_info"

    def test_exception_info_in_extra(self):
        actions = [{"command": "test", "tool_call_id": "call_1"}]
        outputs = [{"output": "", "returncode": 1, "exception_info": "Error occurred", "extra": {"detail": "more"}}]
        result = format_toolcall_observation_messages(
            actions=actions, outputs=outputs, observation_template="{{ output.output }}"
        )
        assert result[0]["extra"]["exception_info"] == "Error occurred"
        assert result[0]["extra"]["detail"] == "more"


class TestBashTool:
    def test_bash_tool_structure(self):
        assert BASH_TOOL["type"] == "function"
        assert BASH_TOOL["function"]["name"] == "bash"
        assert "command" in BASH_TOOL["function"]["parameters"]["properties"]
        assert "command" in BASH_TOOL["function"]["parameters"]["required"]

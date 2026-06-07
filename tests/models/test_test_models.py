import logging
import time

import pytest

import minisweagent.models
from minisweagent.exceptions import FormatError
from minisweagent.models.test_models import (
    DeterministicModel,
    DeterministicModelConfig,
    DeterministicResponseAPIToolcallModel,
    DeterministicResponseAPIToolcallModelConfig,
    DeterministicToolcallModel,
    DeterministicToolcallModelConfig,
    make_output,
    make_response_api_output,
    make_toolcall_output,
)


def test_basic_functionality_and_cost_tracking(reset_global_stats):
    """Test basic model functionality, cost tracking, and default configuration."""
    model = DeterministicModel(
        outputs=[
            make_output("```mswea_bash_command\necho hello\n```", [{"command": "echo hello"}]),
            make_output("```mswea_bash_command\necho world\n```", [{"command": "echo world"}]),
        ]
    )

    # Test first call with defaults
    result = model.query([{"role": "user", "content": "test"}])
    assert result["content"] == "```mswea_bash_command\necho hello\n```"
    assert result["extra"]["actions"] == [{"command": "echo hello"}]
    assert minisweagent.models.GLOBAL_MODEL_STATS.n_calls == 1
    assert minisweagent.models.GLOBAL_MODEL_STATS.cost == 1.0

    # Test second call and sequential outputs
    result = model.query([{"role": "user", "content": "test"}])
    assert result["content"] == "```mswea_bash_command\necho world\n```"
    assert result["extra"]["actions"] == [{"command": "echo world"}]
    assert minisweagent.models.GLOBAL_MODEL_STATS.n_calls == 2
    assert minisweagent.models.GLOBAL_MODEL_STATS.cost == 2.0


def test_custom_cost_and_multiple_models(reset_global_stats):
    """Test custom cost configuration and global tracking across multiple models."""
    model1 = DeterministicModel(
        outputs=[make_output("```mswea_bash_command\necho r1\n```", [{"command": "echo r1"}])], cost_per_call=2.5
    )
    model2 = DeterministicModel(
        outputs=[make_output("```mswea_bash_command\necho r2\n```", [{"command": "echo r2"}])], cost_per_call=3.0
    )

    result1 = model1.query([{"role": "user", "content": "test"}])
    assert result1["content"] == "```mswea_bash_command\necho r1\n```"
    assert minisweagent.models.GLOBAL_MODEL_STATS.cost == 2.5

    result2 = model2.query([{"role": "user", "content": "test"}])
    assert result2["content"] == "```mswea_bash_command\necho r2\n```"
    assert minisweagent.models.GLOBAL_MODEL_STATS.cost == 5.5
    assert minisweagent.models.GLOBAL_MODEL_STATS.n_calls == 2


def test_config_dataclass():
    """Test DeterministicModelConfig with custom values."""
    config = DeterministicModelConfig(
        outputs=[make_output("Test", [{"command": "test"}])], model_name="custom", cost_per_call=5.0
    )

    assert config.cost_per_call == 5.0
    assert config.model_name == "custom"

    model = DeterministicModel(**config.model_dump())
    assert model.config.cost_per_call == 5.0


def test_sleep_and_warning_commands(caplog):
    """Test special /sleep and /warning command handling."""
    # Test sleep command - processes sleep then returns actual output (counts as 1 call)
    model = DeterministicModel(
        outputs=[
            make_output("", [{"command": "/sleep 0.1"}]),
            make_output("```mswea_bash_command\necho after_sleep\n```", [{"command": "echo after_sleep"}]),
        ]
    )
    start_time = time.time()
    result = model.query([{"role": "user", "content": "test"}])
    assert result["content"] == "```mswea_bash_command\necho after_sleep\n```"
    assert time.time() - start_time >= 0.1

    # Test warning command - processes warning then returns actual output (counts as 1 call)
    model2 = DeterministicModel(
        outputs=[
            make_output("", [{"command": "/warning Test message"}]),
            make_output("```mswea_bash_command\necho after_warning\n```", [{"command": "echo after_warning"}]),
        ]
    )
    with caplog.at_level(logging.WARNING):
        result2 = model2.query([{"role": "user", "content": "test"}])
        assert result2["content"] == "```mswea_bash_command\necho after_warning\n```"
    assert "Test message" in caplog.text


def test_raise_exception():
    """Test {"raise": Exception(...)} raises the exception."""
    model = DeterministicModel(outputs=[make_output("", [{"raise": FormatError()}])])
    with pytest.raises(FormatError):
        model.query([{"role": "user", "content": "test"}])


def test_toolcall_model_basic(reset_global_stats):
    """Test DeterministicToolcallModel basic functionality."""
    tool_calls = [
        {"id": "call_123", "type": "function", "function": {"name": "bash", "arguments": '{"command": "ls"}'}}
    ]
    actions = [{"command": "ls", "tool_call_id": "call_123"}]

    model = DeterministicToolcallModel(
        outputs=[make_toolcall_output(None, tool_calls, actions)],
    )

    result = model.query([{"role": "user", "content": "list files"}])
    assert result["tool_calls"] == tool_calls
    assert result["extra"]["actions"] == actions
    assert minisweagent.models.GLOBAL_MODEL_STATS.n_calls == 1


def test_toolcall_model_format_observation(reset_global_stats):
    """Test DeterministicToolcallModel formats observations as tool results."""
    tool_calls = [
        {"id": "call_456", "type": "function", "function": {"name": "bash", "arguments": '{"command": "pwd"}'}}
    ]
    actions = [{"command": "pwd", "tool_call_id": "call_456"}]

    model = DeterministicToolcallModel(outputs=[make_toolcall_output(None, tool_calls, actions)])

    result = model.query([{"role": "user", "content": "test"}])
    outputs = [{"output": "/home/user", "returncode": 0, "exception_info": ""}]
    obs_messages = model.format_observation_messages(result, outputs)

    assert len(obs_messages) == 1
    assert obs_messages[0]["role"] == "tool"
    assert obs_messages[0]["tool_call_id"] == "call_456"
    assert "/home/user" in obs_messages[0]["content"]


def test_toolcall_config():
    """Test DeterministicToolcallModelConfig with custom values."""
    config = DeterministicToolcallModelConfig(
        outputs=[make_toolcall_output(None, [], [])], model_name="custom_toolcall", cost_per_call=2.0
    )

    assert config.cost_per_call == 2.0
    assert config.model_name == "custom_toolcall"

    model = DeterministicToolcallModel(**config.model_dump())
    assert model.config.cost_per_call == 2.0


def test_response_api_model_basic(reset_global_stats):
    """Test DeterministicResponseAPIToolcallModel basic functionality."""
    actions = [{"command": "ls", "tool_call_id": "call_resp_123"}]

    model = DeterministicResponseAPIToolcallModel(
        outputs=[make_response_api_output("I'll list files", actions)],
    )

    result = model.query([{"role": "user", "content": "list files"}])
    assert result["object"] == "response"
    assert result["extra"]["actions"] == actions
    assert minisweagent.models.GLOBAL_MODEL_STATS.n_calls == 1
    # Check output structure
    assert len(result["output"]) == 2  # message + function_call
    assert result["output"][0]["type"] == "message"
    assert result["output"][1]["type"] == "function_call"
    assert result["output"][1]["call_id"] == "call_resp_123"


def test_response_api_model_format_observation(reset_global_stats):
    """Test DeterministicResponseAPIToolcallModel formats observations as function_call_output."""
    actions = [{"command": "pwd", "tool_call_id": "call_resp_456"}]

    model = DeterministicResponseAPIToolcallModel(outputs=[make_response_api_output(None, actions)])

    result = model.query([{"role": "user", "content": "test"}])
    outputs = [{"output": "/home/user", "returncode": 0, "exception_info": ""}]
    obs_messages = model.format_observation_messages(result, outputs)

    assert len(obs_messages) == 1
    assert obs_messages[0]["type"] == "function_call_output"
    assert obs_messages[0]["call_id"] == "call_resp_456"
    assert "/home/user" in obs_messages[0]["output"]


def test_response_api_model_format_message():
    """Test DeterministicResponseAPIToolcallModel formats messages in Responses API format."""
    model = DeterministicResponseAPIToolcallModel(outputs=[])

    msg = model.format_message(role="user", content="Hello")
    assert msg["type"] == "message"
    assert msg["role"] == "user"
    assert msg["content"] == [{"type": "input_text", "text": "Hello"}]


def test_response_api_config():
    """Test DeterministicResponseAPIToolcallModelConfig with custom values."""
    config = DeterministicResponseAPIToolcallModelConfig(
        outputs=[make_response_api_output(None, [])], model_name="custom_response_api", cost_per_call=3.0
    )

    assert config.cost_per_call == 3.0
    assert config.model_name == "custom_response_api"

    model = DeterministicResponseAPIToolcallModel(**config.model_dump())
    assert model.config.cost_per_call == 3.0

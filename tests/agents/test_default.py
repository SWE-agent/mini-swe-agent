from pathlib import Path

import pytest
import yaml

from minisweagent.agents.default import DefaultAgent
from minisweagent.environments.local import LocalEnvironment
from minisweagent.models.test_models import (
    DeterministicModel,
    DeterministicResponseAPIToolcallModel,
    DeterministicToolcallModel,
    make_output,
    make_response_api_output,
    make_toolcall_output,
)

# --- Helper functions to abstract message format differences ---


def get_text(msg: dict) -> str:
    """Extract text content from a message regardless of format."""
    content = msg.get("content")
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list) and content:
        return content[0].get("text", "")
    return ""


def get_observation_text(msg: dict) -> str:
    """Extract observation text from a message (handles all formats)."""
    if msg.get("type") == "function_call_output":
        return msg.get("output", "")
    return get_text(msg)


def is_assistant_message(msg: dict) -> bool:
    """Check if message is an assistant/response message."""
    return msg.get("role") == "assistant" or msg.get("object") == "response"


def is_observation_message(msg: dict) -> bool:
    """Check if message is an observation message."""
    if msg.get("type") == "function_call_output":
        return True
    if msg.get("role") == "tool":
        return True
    if msg.get("role") == "user" and "returncode" in get_text(msg):
        return True
    return False


# --- Fixtures ---


@pytest.fixture
def default_config():
    """Load default agent config from config/default.yaml"""
    config_path = Path("src/minisweagent/config/default.yaml")
    with open(config_path) as f:
        config = yaml.safe_load(f)
    return config["agent"]


@pytest.fixture
def toolcall_config():
    """Load toolcall agent config from config/mini.yaml"""
    config_path = Path("src/minisweagent/config/mini.yaml")
    with open(config_path) as f:
        config = yaml.safe_load(f)
    return config["agent"]


def make_text_model(outputs_spec: list[tuple[str, list[dict]]], **kwargs) -> DeterministicModel:
    """Create a DeterministicModel from a list of (content, actions) tuples."""
    return DeterministicModel(outputs=[make_output(content, actions) for content, actions in outputs_spec], **kwargs)


def make_tc_model(outputs_spec: list[tuple[str, list[dict]]], **kwargs) -> DeterministicToolcallModel:
    """Create a DeterministicToolcallModel from a list of (content, actions) tuples."""
    outputs = []
    for i, (content, actions) in enumerate(outputs_spec):
        tc_actions = []
        tool_calls = []
        for j, action in enumerate(actions):
            tool_call_id = f"call_{i}_{j}"
            tc_actions.append({"command": action["command"], "tool_call_id": tool_call_id})
            tool_calls.append(
                {
                    "id": tool_call_id,
                    "type": "function",
                    "function": {"name": "bash", "arguments": f'{{"command": "{action["command"]}"}}'},
                }
            )
        outputs.append(make_toolcall_output(content, tool_calls, tc_actions))
    return DeterministicToolcallModel(outputs=outputs, **kwargs)


def make_response_api_model(
    outputs_spec: list[tuple[str, list[dict]]], **kwargs
) -> DeterministicResponseAPIToolcallModel:
    """Create a DeterministicResponseAPIToolcallModel from a list of (content, actions) tuples."""
    outputs = []
    for i, (content, actions) in enumerate(outputs_spec):
        api_actions = []
        for j, action in enumerate(actions):
            tool_call_id = f"call_resp_{i}_{j}"
            api_actions.append({"command": action["command"], "tool_call_id": tool_call_id})
        outputs.append(make_response_api_output(content, api_actions))
    return DeterministicResponseAPIToolcallModel(outputs=outputs, **kwargs)


@pytest.fixture(params=["text", "toolcall", "response_api"])
def model_factory(request, default_config, toolcall_config):
    """Parametrized fixture that returns (factory_fn, config) for all three model types."""
    if request.param == "text":
        return make_text_model, default_config
    elif request.param == "toolcall":
        return make_tc_model, toolcall_config
    else:  # response_api
        return make_response_api_model, toolcall_config


# --- Tests ---


def test_successful_completion(model_factory):
    """Test agent completes successfully when COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT is encountered."""
    factory, config = model_factory
    agent = DefaultAgent(
        model=factory(
            [
                ("I'll echo a message", [{"command": "echo 'hello world'"}]),
                (
                    "Now finishing",
                    [{"command": "echo 'COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT'\necho 'Task completed successfully'"}],
                ),
            ]
        ),
        env=LocalEnvironment(),
        **config,
    )

    info = agent.run("Echo hello world then finish")
    assert info["exit_status"] == "Submitted"
    assert info["submission"] == "Task completed successfully\n"
    assert agent.n_calls == 2


def test_step_limit_enforcement(model_factory):
    """Test agent stops when step limit is reached."""
    factory, config = model_factory
    agent = DefaultAgent(
        model=factory(
            [
                ("First command", [{"command": "echo 'step1'"}]),
                ("Second command", [{"command": "echo 'step2'"}]),
            ]
        ),
        env=LocalEnvironment(),
        **{**config, "step_limit": 1},
    )

    info = agent.run("Run multiple commands")
    assert info["exit_status"] == "LimitsExceeded"
    assert agent.n_calls == 1


def test_cost_limit_enforcement(model_factory):
    """Test agent stops when cost limit is reached."""
    factory, config = model_factory
    agent = DefaultAgent(
        model=factory([("Test", [{"command": "echo 'test'"}])]),
        env=LocalEnvironment(),
        **{**config, "cost_limit": 0.5},
    )

    info = agent.run("Test cost limit")
    assert info["exit_status"] == "LimitsExceeded"


def test_timeout_handling(model_factory):
    """Test agent handles command timeouts properly."""
    factory, config = model_factory
    agent = DefaultAgent(
        model=factory(
            [
                ("Long sleep", [{"command": "sleep 5"}]),  # This will timeout
                ("Quick finish", [{"command": "echo 'COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT'\necho 'recovered'"}]),
            ]
        ),
        env=LocalEnvironment(timeout=1),  # Very short timeout
        **config,
    )

    info = agent.run("Test timeout handling")
    assert info["exit_status"] == "Submitted"
    assert info["submission"] == "recovered\n"
    # Should have timeout error message in observation
    timed_out = [msg for msg in agent.messages if "timed out" in get_observation_text(msg)]
    assert len(timed_out) == 1


def test_timeout_captures_partial_output(model_factory):
    """Test that timeout error captures partial output from commands that produce output before timing out."""
    factory, config = model_factory
    num1, num2 = 111, 9
    calculation_command = f"echo $(({num1}*{num2})); sleep 10"
    expected_output = str(num1 * num2)
    agent = DefaultAgent(
        model=factory(
            [
                ("Output then sleep", [{"command": calculation_command}]),
                ("Quick finish", [{"command": "echo 'COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT'\necho 'recovered'"}]),
            ]
        ),
        env=LocalEnvironment(timeout=1),
        **config,
    )
    info = agent.run("Test timeout with partial output")
    assert info["exit_status"] == "Submitted"
    assert info["submission"] == "recovered\n"
    timed_out = [msg for msg in agent.messages if "timed out" in get_observation_text(msg)]
    assert len(timed_out) == 1
    assert expected_output in get_observation_text(timed_out[0])


def test_multiple_steps_before_completion(model_factory):
    """Test agent can handle multiple steps before finding completion signal."""
    factory, config = model_factory
    agent = DefaultAgent(
        model=factory(
            [
                ("Step 1", [{"command": "echo 'first'"}]),
                ("Step 2", [{"command": "echo 'second'"}]),
                ("Step 3", [{"command": "echo 'third'"}]),
                (
                    "Final step",
                    [{"command": "echo 'COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT'\necho 'completed all steps'"}],
                ),
            ]
        ),
        env=LocalEnvironment(),
        **{**config, "cost_limit": 5.0},  # Increase cost limit to allow all 4 calls
    )

    info = agent.run("Multi-step task")
    assert info["exit_status"] == "Submitted"
    assert info["submission"] == "completed all steps\n"
    assert agent.n_calls == 4


def test_custom_config(model_factory):
    """Test agent works with custom configuration."""
    factory, config = model_factory
    agent = DefaultAgent(
        model=factory(
            [
                (
                    "Test response",
                    [{"command": "echo 'COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT'\necho 'custom config works'"}],
                )
            ]
        ),
        env=LocalEnvironment(),
        **{
            **config,
            "system_template": "You are a test assistant.",
            "instance_template": "Task: {{task}}. Return bash command.",
            "step_limit": 2,
            "cost_limit": 1.0,
        },
    )

    info = agent.run("Test custom config")
    assert info["exit_status"] == "Submitted"
    assert info["submission"] == "custom config works\n"
    assert get_text(agent.messages[0]) == "You are a test assistant."
    assert "Test custom config" in get_text(agent.messages[1])


def test_render_template_model_stats(model_factory):
    """Test that render_template has access to n_model_calls and model_cost from agent."""
    factory, config = model_factory
    agent = DefaultAgent(
        model=factory(
            [
                ("Test 1", [{"command": "echo 'test1'"}]),
                ("Test 2", [{"command": "echo 'test2'"}]),
            ]
        ),
        env=LocalEnvironment(),
        **config,
    )

    # Make some calls through the agent to generate stats
    agent.add_messages({"role": "system", "content": "test"}, {"role": "user", "content": "test"})
    agent.query()
    agent.query()

    # Test template rendering with agent stats
    template = "Calls: {{n_model_calls}}, Cost: {{model_cost}}"
    assert agent._render_template(template) == "Calls: 2, Cost: 2.0"


def test_messages_include_timestamps(model_factory):
    """Test that assistant and observation messages include timestamps."""
    factory, config = model_factory
    agent = DefaultAgent(
        model=factory(
            [
                ("Response 1", [{"command": "echo 'test1'"}]),
                ("Response 2", [{"command": "echo 'COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT'\necho 'done'"}]),
            ]
        ),
        env=LocalEnvironment(),
        **config,
    )

    agent.run("Test timestamps")

    # Assistant messages should have timestamps
    assistant_msgs = [msg for msg in agent.messages if is_assistant_message(msg)]
    assert all("timestamp" in msg.get("extra", {}) for msg in assistant_msgs)
    # Timestamps should be numeric (floats from time.time())
    all_timestamped = [msg for msg in agent.messages if "timestamp" in msg.get("extra", {})]
    assert all(isinstance(msg["extra"]["timestamp"], float) for msg in all_timestamped)


def test_message_history_tracking(model_factory):
    """Test that messages are properly added and tracked."""
    factory, config = model_factory
    agent = DefaultAgent(
        model=factory(
            [
                ("Response 1", [{"command": "echo 'test1'"}]),
                ("Response 2", [{"command": "echo 'COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT'\necho 'done'"}]),
            ]
        ),
        env=LocalEnvironment(),
        **config,
    )

    info = agent.run("Track messages")
    assert info["exit_status"] == "Submitted"
    assert info["submission"] == "done\n"

    # Should have 6 messages: system, user, assistant, observation, assistant, exit
    assert len(agent.messages) == 6
    # First two are system and user
    assert get_text(agent.messages[0])  # system has content
    assert get_text(agent.messages[1])  # user has content
    # Third is assistant response
    assert is_assistant_message(agent.messages[2])
    # Fourth is observation
    assert is_observation_message(agent.messages[3])
    # Fifth is assistant response
    assert is_assistant_message(agent.messages[4])


def test_step_adds_messages(model_factory):
    """Test that step adds assistant and observation messages."""
    factory, config = model_factory
    agent = DefaultAgent(
        model=factory([("Test command", [{"command": "echo 'hello'"}])]),
        env=LocalEnvironment(),
        **config,
    )

    agent.add_messages({"role": "system", "content": "system message"})
    agent.add_messages({"role": "user", "content": "user message"})

    initial_count = len(agent.messages)
    agent.step()

    # step() should add assistant message + observation message
    assert len(agent.messages) == initial_count + 2
    assert is_assistant_message(agent.messages[-2])
    assert agent.messages[-2]["extra"]["actions"][0]["command"] == "echo 'hello'"
    assert is_observation_message(agent.messages[-1])
    assert "returncode" in get_observation_text(agent.messages[-1])


def test_observations_captured(model_factory):
    """Test intermediate outputs are captured correctly."""
    factory, config = model_factory
    agent = DefaultAgent(
        model=factory(
            [
                ("Step 1", [{"command": "echo 'first'"}]),
                ("Step 2", [{"command": "echo 'second'"}]),
                ("Final", [{"command": "echo 'COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT'\necho 'done'"}]),
            ]
        ),
        env=LocalEnvironment(),
        **{**config, "cost_limit": 5.0},
    )

    agent.run("Multi-step task")
    observations = [get_observation_text(msg) for msg in agent.messages if is_observation_message(msg)]
    assert len(observations) == 2
    assert "first" in observations[0]
    assert "second" in observations[1]


def test_empty_actions_handling(model_factory):
    """Test agent handles empty actions (continues without error)."""
    factory, config = model_factory
    agent = DefaultAgent(
        model=factory(
            [
                ("No actions here", []),  # Empty actions list
                ("Now with action", [{"command": "echo 'COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT'\necho 'done'"}]),
            ]
        ),
        env=LocalEnvironment(),
        **config,
    )

    info = agent.run("Test empty actions")
    assert info["exit_status"] == "Submitted"
    assert info["submission"] == "done\n"
    assert agent.n_calls == 2

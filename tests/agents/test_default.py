from pathlib import Path

import pytest
import yaml

from minisweagent.agents.default import DefaultAgent
from minisweagent.environments.local import LocalEnvironment
from minisweagent.models.test_models import DeterministicModel


@pytest.fixture
def default_config():
    """Load default agent config from config/default.yaml"""
    config_path = Path("src/minisweagent/config/default.yaml")
    with open(config_path) as f:
        config = yaml.safe_load(f)
    return config["agent"]


def test_successful_completion(default_config):
    """Test agent completes successfully when COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT is encountered."""
    agent = DefaultAgent(
        model=DeterministicModel(
            outputs=[
                "I'll echo a message\n```bash\necho 'hello world'\n```",
                "Now finishing\n```bash\necho 'Task completed successfully'\necho 'COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT'\n```",
            ]
        ),
        env=LocalEnvironment(),
        **default_config,
    )

    info = agent.run("Echo hello world then finish")
    assert info["exit_status"] == "Submitted"
    assert info["submission"] == "Task completed successfully\n"
    assert agent.model.n_calls == 2
    assert len(agent.messages) == 6  # system, user, assistant, user, assistant, user


def test_step_limit_enforcement(default_config):
    """Test agent stops when step limit is reached."""
    agent = DefaultAgent(
        model=DeterministicModel(
            outputs=["First command\n```bash\necho 'step1'\n```", "Second command\n```bash\necho 'step2'\n```"]
        ),
        env=LocalEnvironment(),
        **{**default_config, "step_limit": 1},
    )

    info = agent.run("Run multiple commands")
    assert info["exit_status"] == "LimitsExceeded"
    assert agent.model.n_calls == 1


def test_cost_limit_enforcement(default_config):
    """Test agent stops when cost limit is reached."""
    model = DeterministicModel(outputs=["```bash\necho 'test'\n```"])

    agent = DefaultAgent(
        model=model,
        env=LocalEnvironment(),
        **{**default_config, "cost_limit": 0.5},
    )

    info = agent.run("Test cost limit")
    assert info["exit_status"] == "LimitsExceeded"


def test_format_error_handling(default_config):
    """Test agent handles malformed action formats properly."""
    agent = DefaultAgent(
        model=DeterministicModel(
            outputs=[
                "No code blocks here",
                "Multiple blocks\n```bash\necho 'first'\n```\n```bash\necho 'second'\n```",
                "Now correct\n```bash\necho 'done'\necho 'COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT'\n```",
            ]
        ),
        env=LocalEnvironment(),
        **default_config,
    )

    info = agent.run("Test format errors")
    assert info["exit_status"] == "Submitted"
    assert info["submission"] == "done\n"
    assert agent.model.n_calls == 3
    # Should have error messages in conversation
    assert (
        len([msg for msg in agent.messages if "Please always provide EXACTLY ONE action" in msg.get("content", "")])
        == 2
    )


def test_timeout_handling(default_config):
    """Test agent handles command timeouts properly."""
    agent = DefaultAgent(
        model=DeterministicModel(
            outputs=[
                "Long sleep\n```bash\nsleep 5\n```",  # This will timeout
                "Quick finish\n```bash\necho 'recovered'\necho 'COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT'\n```",
            ]
        ),
        env=LocalEnvironment(timeout=1),  # Very short timeout
        **default_config,
    )

    info = agent.run("Test timeout handling")
    assert info["exit_status"] == "Submitted"
    assert info["submission"] == "recovered\n"
    # Should have timeout error message
    assert len([msg for msg in agent.messages if "timed out" in msg.get("content", "")]) == 1


def test_timeout_captures_partial_output(default_config):
    """Test that timeout error captures partial output from commands that produce output before timing out."""
    num1, num2 = 111, 9
    calculation_command = f"echo $(({num1}*{num2})); sleep 10"
    expected_output = str(num1 * num2)
    agent = DefaultAgent(
        model=DeterministicModel(
            outputs=[
                f"Output then sleep\n```bash\n{calculation_command}\n```",
                "Quick finish\n```bash\necho 'recovered'\necho 'COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT'\n```",
            ]
        ),
        env=LocalEnvironment(timeout=1),
        **default_config,
    )
    info = agent.run("Test timeout with partial output")
    assert info["exit_status"] == "Submitted"
    assert info["submission"] == "recovered\n"  # final output should be `recovered` from the last command
    timed_out_messages = [msg for msg in agent.messages if "timed out" in msg.get("content", "")]
    assert len(timed_out_messages) == 1
    assert expected_output in timed_out_messages[0]["content"]  # ensure timed out output is still captured


def test_model_parse_action_success(default_config):
    """Test action parsing works correctly for valid formats (now on model)."""
    model = DeterministicModel(outputs=[])

    # Test different valid formats
    assert model.parse_action("```bash\necho 'test'\n```") == "echo 'test'"
    assert model.parse_action("```bash\nls -la\n```") == "ls -la"
    assert model.parse_action("Some text\n```bash\necho 'hello'\n```\nMore text") == "echo 'hello'"


def test_model_parse_action_failures(default_config):
    """Test action parsing raises appropriate exceptions for invalid formats (now on model)."""
    from minisweagent.exceptions import InterruptAgentFlow

    model = DeterministicModel(outputs=[])

    # No code blocks
    with pytest.raises(InterruptAgentFlow):
        model.parse_action("No code blocks here")

    # Multiple code blocks
    with pytest.raises(InterruptAgentFlow):
        model.parse_action("```bash\necho 'first'\n```\n```bash\necho 'second'\n```")

    # Code block without bash language specifier
    with pytest.raises(InterruptAgentFlow):
        model.parse_action("```\nls -la\n```")


def test_message_history_tracking(default_config):
    """Test that messages are properly added and tracked."""
    agent = DefaultAgent(
        model=DeterministicModel(
            outputs=[
                "Response 1\n```bash\necho 'test1'\n```",
                "Response 2\n```bash\necho 'done'\necho 'COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT'\n```",
            ]
        ),
        env=LocalEnvironment(),
        **default_config,
    )

    info = agent.run("Track messages")
    assert info["exit_status"] == "Submitted"
    assert info["submission"] == "done\n"

    # After completion should have full conversation (exit instead of user at end)
    assert len(agent.messages) == 6
    assert [msg["role"] for msg in agent.messages] == ["system", "user", "assistant", "user", "assistant", "exit"]


def test_multiple_steps_before_completion(default_config):
    """Test agent can handle multiple steps before finding completion signal."""
    agent = DefaultAgent(
        model=DeterministicModel(
            outputs=[
                "Step 1\n```bash\necho 'first'\n```",
                "Step 2\n```bash\necho 'second'\n```",
                "Step 3\n```bash\necho 'third'\n```",
                "Final step\n```bash\necho 'completed all steps'\necho 'COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT'\n```",
            ]
        ),
        env=LocalEnvironment(),
        **{**default_config, "cost_limit": 5.0},  # Increase cost limit to allow all 4 calls (4.0 total cost)
    )

    info = agent.run("Multi-step task")
    assert info["exit_status"] == "Submitted"
    assert info["submission"] == "completed all steps\n"
    assert agent.model.n_calls == 4

    # Check that all intermediate outputs are captured (final step doesn't get observation due to termination)
    observations = [
        msg["content"] for msg in agent.messages if msg["role"] == "user" and "<returncode>" in msg["content"]
    ]
    assert len(observations) == 3
    assert "first" in observations[0]
    assert "second" in observations[1]
    assert "third" in observations[2]


def test_custom_config(default_config):
    """Test agent works with custom configuration."""
    agent = DefaultAgent(
        model=DeterministicModel(
            outputs=[
                "Test response\n```bash\necho 'custom config works'\necho 'COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT'\n```"
            ]
        ),
        env=LocalEnvironment(),
        **{
            **default_config,
            "system_template": "You are a test assistant.",
            "instance_template": "Task: {{task}}. Return bash command.",
            "step_limit": 2,
            "cost_limit": 1.0,
        },
    )

    info = agent.run("Test custom config")
    assert info["exit_status"] == "Submitted"
    assert info["submission"] == "custom config works\n"
    assert agent.messages[0]["content"] == "You are a test assistant."
    assert "Test custom config" in agent.messages[1]["content"]


def test_render_template_model_stats(default_config):
    """Test that render_template has access to n_model_calls and model_cost from agent."""
    agent = DefaultAgent(
        model=DeterministicModel(outputs=["```bash\necho 'test1'\n```", "```bash\necho 'test2'\n```"]),
        env=LocalEnvironment(),
        **default_config,
    )

    # Make some calls through the agent to generate stats
    agent.add_messages({"role": "system", "content": "test"}, {"role": "user", "content": "test"})
    agent.query()
    agent.query()

    # Test template rendering with agent stats
    template = "Calls: {{n_model_calls}}, Cost: {{model_cost}}"
    assert agent._render_template(template) == "Calls: 2, Cost: 2.0"


def test_messages_include_timestamps(default_config):
    """Test that assistant and observation messages include timestamps."""
    agent = DefaultAgent(
        model=DeterministicModel(
            outputs=[
                "Response 1\n```bash\necho 'test1'\n```",
                "Response 2\n```bash\necho 'done'\necho 'COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT'\n```",
            ]
        ),
        env=LocalEnvironment(),
        **default_config,
    )

    agent.run("Test timestamps")

    # Assistant and observation messages should have timestamps (system/user from agent don't have them)
    assistant_msgs = [msg for msg in agent.messages if msg["role"] == "assistant"]
    obs_msgs = [msg for msg in agent.messages if msg["role"] == "user" and "<returncode>" in msg.get("content", "")]
    assert all("timestamp" in msg.get("extra", {}) for msg in assistant_msgs)
    assert all("timestamp" in msg.get("extra", {}) for msg in obs_msgs)
    # Timestamps should be numeric (floats from time.time())
    all_timestamped = [msg for msg in agent.messages if "timestamp" in msg.get("extra", {})]
    assert all(isinstance(msg["extra"]["timestamp"], float) for msg in all_timestamped)
    # Timestamps should be monotonically increasing (in the order they appear in messages)
    timestamps = [msg["extra"]["timestamp"] for msg in all_timestamped]
    assert timestamps == sorted(timestamps)


def test_step_adds_messages(default_config):
    """Test that step adds assistant and observation messages."""
    agent = DefaultAgent(
        model=DeterministicModel(outputs=["Test command\n```bash\necho 'hello'\n```"]),
        env=LocalEnvironment(),
        **default_config,
    )

    agent.add_messages({"role": "system", "content": "system message"})
    agent.add_messages({"role": "user", "content": "user message"})

    initial_count = len(agent.messages)
    agent.step()

    # step() should add assistant message + observation message
    assert len(agent.messages) == initial_count + 2
    assert agent.messages[-2]["role"] == "assistant"
    assert agent.messages[-2]["extra"]["action"] == "echo 'hello'"
    assert agent.messages[-1]["role"] == "user"
    assert "<returncode>" in agent.messages[-1]["content"]

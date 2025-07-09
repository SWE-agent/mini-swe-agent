import pytest

from microswea.agents.default import DefaultAgent, NonTerminatingException
from microswea.environments.local import LocalEnvironment
from microswea.models.test_models import DeterministicModel


def test_successful_completion():
    """Test agent completes successfully when MICRO_SWE_AGENT_FINAL_OUTPUT is encountered."""
    agent = DefaultAgent(
        model=DeterministicModel(
            outputs=[
                "I'll echo a message\n```bash\necho 'hello world'\n```",
                "Now finishing\n```bash\necho 'MICRO_SWE_AGENT_FINAL_OUTPUT'\necho 'Task completed successfully'\n```",
            ]
        ),
        env=LocalEnvironment(),
    )

    exit_status, result = agent.run("Echo hello world then finish")
    assert exit_status == "Submitted"
    assert result == "Task completed successfully"
    assert agent.model.n_calls == 2
    assert len(agent.messages) == 6  # system, user, assistant, user, assistant, user


def test_step_limit_enforcement():
    """Test agent stops when step limit is reached."""
    agent = DefaultAgent(
        model=DeterministicModel(
            outputs=["First command\n```bash\necho 'step1'\n```", "Second command\n```bash\necho 'step2'\n```"]
        ),
        env=LocalEnvironment(),
        step_limit=1,
    )

    exit_status, result = agent.run("Run multiple commands")
    assert exit_status == "LimitsExceeded"
    assert agent.model.n_calls == 1


def test_cost_limit_enforcement():
    """Test agent stops when cost limit is reached."""
    model = DeterministicModel(outputs=["```bash\necho 'test'\n```"])
    model.cost = 1.0

    agent = DefaultAgent(
        model=model,
        env=LocalEnvironment(),
        cost_limit=0.5,
    )

    exit_status, result = agent.run("Test cost limit")
    assert exit_status == "LimitsExceeded"


def test_format_error_handling():
    """Test agent handles malformed action formats properly."""
    agent = DefaultAgent(
        model=DeterministicModel(
            outputs=[
                "No code blocks here",
                "Multiple blocks\n```bash\necho 'first'\n```\n```bash\necho 'second'\n```",
                "Now correct\n```bash\necho 'MICRO_SWE_AGENT_FINAL_OUTPUT'\necho 'done'\n```",
            ]
        ),
        env=LocalEnvironment(),
    )

    exit_status, result = agent.run("Test format errors")
    assert exit_status == "Submitted"
    assert result == "done"
    assert agent.model.n_calls == 3
    # Should have error messages in conversation
    assert (
        len([msg for msg in agent.messages if "Please always provide EXACTLY ONE action" in msg.get("content", "")])
        == 2
    )


def test_timeout_handling():
    """Test agent handles command timeouts properly."""
    agent = DefaultAgent(
        model=DeterministicModel(
            outputs=[
                "Long sleep\n```bash\nsleep 5\n```",  # This will timeout
                "Quick finish\n```bash\necho 'MICRO_SWE_AGENT_FINAL_OUTPUT'\necho 'recovered'\n```",
            ]
        ),
        env=LocalEnvironment(timeout=1),  # Very short timeout
    )

    exit_status, result = agent.run("Test timeout handling")
    assert exit_status == "Submitted"
    assert result == "recovered"
    # Should have timeout error message
    assert len([msg for msg in agent.messages if "timed out" in msg.get("content", "")]) == 1


def test_parse_action_success():
    """Test action parsing works correctly for valid formats."""
    agent = DefaultAgent(
        model=DeterministicModel(outputs=[]),
        env=LocalEnvironment(),
    )

    # Test different valid formats
    assert agent.parse_action("```bash\necho 'test'\n```") == "echo 'test'"
    assert agent.parse_action("```\nls -la\n```") == "ls -la"
    assert agent.parse_action("Some text\n```bash\necho 'hello'\n```\nMore text") == "echo 'hello'"


def test_parse_action_failures():
    """Test action parsing raises appropriate exceptions for invalid formats."""
    agent = DefaultAgent(
        model=DeterministicModel(outputs=[]),
        env=LocalEnvironment(),
    )

    # No code blocks
    with pytest.raises(NonTerminatingException):
        agent.parse_action("No code blocks here")

    # Multiple code blocks
    with pytest.raises(NonTerminatingException):
        agent.parse_action("```bash\necho 'first'\n```\n```bash\necho 'second'\n```")


def test_message_history_tracking():
    """Test that messages are properly added and tracked."""
    agent = DefaultAgent(
        model=DeterministicModel(
            outputs=[
                "Response 1\n```bash\necho 'test1'\n```",
                "Response 2\n```bash\necho 'MICRO_SWE_AGENT_FINAL_OUTPUT'\necho 'done'\n```",
            ]
        ),
        env=LocalEnvironment(),
    )

    exit_status, result = agent.run("Track messages")
    assert exit_status == "Submitted"
    assert result == "done"

    # After completion should have full conversation
    assert len(agent.messages) == 6
    assert [msg["role"] for msg in agent.messages] == ["system", "user", "assistant", "user", "assistant", "user"]


def test_multiple_steps_before_completion():
    """Test agent can handle multiple steps before finding completion signal."""
    agent = DefaultAgent(
        model=DeterministicModel(
            outputs=[
                "Step 1\n```bash\necho 'first'\n```",
                "Step 2\n```bash\necho 'second'\n```",
                "Step 3\n```bash\necho 'third'\n```",
                "Final step\n```bash\necho 'MICRO_SWE_AGENT_FINAL_OUTPUT'\necho 'completed all steps'\n```",
            ]
        ),
        env=LocalEnvironment(),
        cost_limit=5.0,  # Increase cost limit to allow all 4 calls (4.0 total cost)
    )

    exit_status, result = agent.run("Multi-step task")
    assert exit_status == "Submitted"
    assert result == "completed all steps"
    assert agent.model.n_calls == 4

    # Check that all intermediate outputs are captured (final step doesn't get observation due to termination)
    observations = [
        msg["content"] for msg in agent.messages if msg["role"] == "user" and "Observation:" in msg["content"]
    ]
    assert len(observations) == 3
    assert "first" in observations[0]
    assert "second" in observations[1]
    assert "third" in observations[2]


def test_custom_config():
    """Test agent works with custom configuration."""
    agent = DefaultAgent(
        model=DeterministicModel(
            outputs=["Test response\n```bash\necho 'MICRO_SWE_AGENT_FINAL_OUTPUT'\necho 'custom config works'\n```"]
        ),
        env=LocalEnvironment(),
        system_template="You are a test assistant.",
        instance_template="Task: {{task}}. Return bash command.",
        step_limit=2,
        cost_limit=1.0,
    )

    exit_status, result = agent.run("Test custom config")
    assert exit_status == "Submitted"
    assert result == "custom config works"
    assert agent.messages[0]["content"] == "You are a test assistant."
    assert "Test custom config" in agent.messages[1]["content"]

import pytest

from minisweagent.agents.default import DefaultAgent, NonTerminatingException
from minisweagent.environments.local import LocalEnvironment
from minisweagent.models.test_models import DeterministicModel


def test_successful_completion():
    """Test agent completes successfully when COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT is encountered."""
    agent = DefaultAgent(
        model=DeterministicModel(
            outputs=[
                "I'll echo a message\n```bash\necho 'hello world'\n```",
                "Now finishing\n```bash\necho 'COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT'\necho 'Task completed successfully'\n```",
            ]
        ),
        env=LocalEnvironment(),
    )

    exit_status, result = agent.run("Echo hello world then finish")
    assert exit_status == "Submitted"
    assert result == "Task completed successfully\n"
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

    exit_status, _ = agent.run("Run multiple commands")
    assert exit_status == "LimitsExceeded"
    assert agent.model.n_calls == 1


def test_cost_limit_enforcement():
    """Test agent stops when cost limit is reached."""
    model = DeterministicModel(outputs=["```bash\necho 'test'\n```"])

    agent = DefaultAgent(
        model=model,
        env=LocalEnvironment(),
        cost_limit=0.5,
    )

    exit_status, _ = agent.run("Test cost limit")
    assert exit_status == "LimitsExceeded"


def test_format_error_handling():
    """Test agent handles malformed action formats properly."""
    agent = DefaultAgent(
        model=DeterministicModel(
            outputs=[
                "No code blocks here",
                "Multiple blocks\n```bash\necho 'first'\n```\n```bash\necho 'second'\n```",
                "Now correct\n```bash\necho 'COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT'\necho 'done'\n```",
            ]
        ),
        env=LocalEnvironment(),
    )

    exit_status, result = agent.run("Test format errors")
    assert exit_status == "Submitted"
    assert result == "done\n"
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
                "Quick finish\n```bash\necho 'COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT'\necho 'recovered'\n```",
            ]
        ),
        env=LocalEnvironment(timeout=1),  # Very short timeout
    )

    exit_status, result = agent.run("Test timeout handling")
    assert exit_status == "Submitted"
    assert result == "recovered\n"
    # Should have timeout error message
    assert len([msg for msg in agent.messages if "timed out" in msg.get("content", "")]) == 1


def test_timeout_captures_partial_output():
    """Test that timeout error captures partial output from commands that produce output before timing out."""
    num1, num2 = 111, 9
    calculation_command = f"echo $(({num1}*{num2})); sleep 10"
    expected_output = str(num1 * num2)
    agent = DefaultAgent(
        model=DeterministicModel(
            outputs=[
                f"Output then sleep\n```bash\n{calculation_command}\n```",
                "Quick finish\n```bash\necho 'COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT'\necho 'recovered'\n```",
            ]
        ),
        env=LocalEnvironment(timeout=1),
    )
    exit_status, result = agent.run("Test timeout with partial output")
    assert exit_status == "Submitted"
    assert result == "recovered\n"  # final output should be `recovered` from the last command
    timed_out_messages = [msg for msg in agent.messages if "timed out" in msg.get("content", "")]
    assert len(timed_out_messages) == 1
    assert expected_output in timed_out_messages[0]["content"]  # ensure timed out output is still captured


def test_parse_action_success():
    """Test action parsing works correctly for valid formats."""
    agent = DefaultAgent(
        model=DeterministicModel(outputs=[]),
        env=LocalEnvironment(),
    )

    # Test different valid formats
    result = agent.parse_action({"content": "```bash\necho 'test'\n```"})
    assert result["action"] == "echo 'test'"
    assert result["content"] == "```bash\necho 'test'\n```"

    result = agent.parse_action({"content": "```bash\nls -la\n```"})
    assert result["action"] == "ls -la"
    assert result["content"] == "```bash\nls -la\n```"

    result = agent.parse_action({"content": "Some text\n```bash\necho 'hello'\n```\nMore text"})
    assert result["action"] == "echo 'hello'"
    assert result["content"] == "Some text\n```bash\necho 'hello'\n```\nMore text"


def test_parse_action_failures():
    """Test action parsing raises appropriate exceptions for invalid formats."""
    agent = DefaultAgent(
        model=DeterministicModel(outputs=[]),
        env=LocalEnvironment(),
    )

    # No code blocks
    with pytest.raises(NonTerminatingException):
        agent.parse_action({"content": "No code blocks here"})

    # Multiple code blocks
    with pytest.raises(NonTerminatingException):
        agent.parse_action({"content": "```bash\necho 'first'\n```\n```bash\necho 'second'\n```"})

    # Code block without bash language specifier
    with pytest.raises(NonTerminatingException):
        agent.parse_action({"content": "```\nls -la\n```"})


def test_message_history_tracking():
    """Test that messages are properly added and tracked."""
    agent = DefaultAgent(
        model=DeterministicModel(
            outputs=[
                "Response 1\n```bash\necho 'test1'\n```",
                "Response 2\n```bash\necho 'COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT'\necho 'done'\n```",
            ]
        ),
        env=LocalEnvironment(),
    )

    exit_status, result = agent.run("Track messages")
    assert exit_status == "Submitted"
    assert result == "done\n"

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
                "Final step\n```bash\necho 'COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT'\necho 'completed all steps'\n```",
            ]
        ),
        env=LocalEnvironment(),
        cost_limit=5.0,  # Increase cost limit to allow all 4 calls (4.0 total cost)
    )

    exit_status, result = agent.run("Multi-step task")
    assert exit_status == "Submitted"
    assert result == "completed all steps\n"
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
            outputs=[
                "Test response\n```bash\necho 'COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT'\necho 'custom config works'\n```"
            ]
        ),
        env=LocalEnvironment(),
        system_template="You are a test assistant.",
        instance_template="Task: {{task}}. Return bash command.",
        step_limit=2,
        cost_limit=1.0,
    )

    exit_status, result = agent.run("Test custom config")
    assert exit_status == "Submitted"
    assert result == "custom config works\n"
    assert agent.messages[0]["content"] == "You are a test assistant."
    assert "Test custom config" in agent.messages[1]["content"]


def test_render_template_model_stats():
    """Test that render_template has access to n_model_calls and model_cost from model."""
    agent = DefaultAgent(
        model=DeterministicModel(outputs=["output1", "output2"]),
        env=LocalEnvironment(),
    )

    # Make some model calls to generate stats
    agent.model.query([])
    agent.model.query([])

    # Test template rendering with model stats
    template = "Calls: {{n_model_calls}}, Cost: {{model_cost}}"
    result = agent.render_template(template)

    assert result == "Calls: 2, Cost: 2.0"


def test_messages_include_timestamps():
    """Test that all messages include timestamps."""
    agent = DefaultAgent(
        model=DeterministicModel(
            outputs=[
                "Response 1\n```bash\necho 'test1'\n```",
                "Response 2\n```bash\necho 'COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT'\necho 'done'\n```",
            ]
        ),
        env=LocalEnvironment(),
    )

    agent.run("Test timestamps")

    # All messages should have timestamps
    assert all("timestamp" in msg for msg in agent.messages)
    # Timestamps should be numeric (floats from time.time())
    assert all(isinstance(msg["timestamp"], float) for msg in agent.messages)
    # Timestamps should be monotonically increasing
    timestamps = [msg["timestamp"] for msg in agent.messages]
    assert timestamps == sorted(timestamps)


def test_step_output_includes_action():
    """Test that step output includes the action that was executed."""
    agent = DefaultAgent(
        model=DeterministicModel(outputs=["Test command\n```bash\necho 'hello'\n```"]),
        env=LocalEnvironment(),
    )

    agent.add_message("system", "system message")
    agent.add_message("user", "user message")

    output = agent.step()

    assert "action" in output
    assert output["action"] == "echo 'hello'"
    assert "output" in output

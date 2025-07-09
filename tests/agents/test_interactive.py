from unittest.mock import patch

from microswea.agents.interactive import InteractiveAgent
from microswea.environments.local import LocalEnvironment
from microswea.models.test_models import DeterministicModel


def test_successful_completion_with_confirmation():
    """Test agent completes successfully when user confirms all actions."""
    with patch("microswea.agents.interactive.console.input", side_effect=[""]):  # Confirm action with Enter
        agent = InteractiveAgent(
            model=DeterministicModel(
                outputs=["Finishing\n```bash\necho 'MICRO_SWE_AGENT_FINAL_OUTPUT'\necho 'completed'\n```"]
            ),
            env=LocalEnvironment(),
            problem_statement="Test completion with confirmation",
        )

        exit_status, result = agent.run()
        assert exit_status == "Submitted"
        assert result == "completed"
        assert agent.model.n_calls == 1


def test_action_rejection_and_recovery():
    """Test agent handles action rejection and can recover."""
    with patch(
        "microswea.agents.interactive.console.input",
        side_effect=[
            "User rejected this action",  # Reject first action
            "",  # Confirm second action
        ],
    ):
        agent = InteractiveAgent(
            model=DeterministicModel(
                outputs=[
                    "First try\n```bash\necho 'first attempt'\n```",
                    "Second try\n```bash\necho 'MICRO_SWE_AGENT_FINAL_OUTPUT'\necho 'recovered'\n```",
                ]
            ),
            env=LocalEnvironment(),
            problem_statement="Test action rejection",
        )

        exit_status, result = agent.run()
        assert exit_status == "Submitted"
        assert result == "recovered"
        assert agent.model.n_calls == 2
        # Should have rejection message in conversation
        rejection_messages = [msg for msg in agent.messages if "User rejected this action" in msg.get("content", "")]
        assert len(rejection_messages) == 1


def test_yolo_mode_activation():
    """Test entering yolo mode disables confirmations."""
    with patch(
        "microswea.agents.interactive.console.input",
        side_effect=[
            "/y",  # Enter yolo mode
            "",  # This should be ignored since yolo mode is on
        ],
    ):
        agent = InteractiveAgent(
            model=DeterministicModel(
                outputs=["Test command\n```bash\necho 'MICRO_SWE_AGENT_FINAL_OUTPUT'\necho 'yolo works'\n```"]
            ),
            env=LocalEnvironment(),
            problem_statement="Test yolo mode",
        )

        exit_status, result = agent.run()
        assert exit_status == "Submitted"
        assert result == "yolo works"
        assert agent.config.confirm_actions is False


def test_yolo_mode_exit():
    """Test exiting yolo mode re-enables confirmations."""
    with patch(
        "microswea.agents.interactive.console.input",
        side_effect=[
            "/y",  # Enter yolo mode
            "/x",  # Exit yolo mode
            "",  # Confirm action (should be required again)
        ],
    ):
        agent = InteractiveAgent(
            model=DeterministicModel(
                outputs=["Test command\n```bash\necho 'MICRO_SWE_AGENT_FINAL_OUTPUT'\necho 'exit yolo works'\n```"]
            ),
            env=LocalEnvironment(),
            problem_statement="Test yolo mode exit",
        )

        exit_status, result = agent.run()
        assert exit_status == "Submitted"
        assert result == "exit yolo works"
        assert agent.config.confirm_actions is True


def test_help_command():
    """Test help command shows help and continues normally."""
    with patch(
        "microswea.agents.interactive.console.input",
        side_effect=[
            "/h",  # Show help
            "",  # Confirm action after help
        ],
    ):
        with patch("microswea.agents.interactive.console.print") as mock_print:
            agent = InteractiveAgent(
                model=DeterministicModel(
                    outputs=["Test help\n```bash\necho 'MICRO_SWE_AGENT_FINAL_OUTPUT'\necho 'help shown'\n```"]
                ),
                env=LocalEnvironment(),
                problem_statement="Test help command",
            )

            exit_status, result = agent.run()
            assert exit_status == "Submitted"
            assert result == "help shown"
            # Check that help was printed
            help_calls = [call for call in mock_print.call_args_list if "/y" in str(call)]
            assert len(help_calls) > 0


def test_whitelisted_actions_skip_confirmation():
    """Test that whitelisted actions don't require confirmation."""
    agent = InteractiveAgent(
        model=DeterministicModel(
            outputs=["Whitelisted\n```bash\necho 'MICRO_SWE_AGENT_FINAL_OUTPUT'\necho 'no confirmation needed'\n```"]
        ),
        env=LocalEnvironment(),
        problem_statement="Test whitelisted actions",
        whitelist_actions=[r"echo.*"],
    )

    # No patch needed - should not ask for confirmation
    exit_status, result = agent.run()
    assert exit_status == "Submitted"
    assert result == "no confirmation needed"


def _test_interruption_helper(interruption_input, expected_message_fragment, problem_statement="Test interruption"):
    """Helper function for testing interruption scenarios."""
    agent = InteractiveAgent(
        model=DeterministicModel(
            outputs=[
                "Initial step\n```bash\necho 'will be interrupted'\n```",
                "Recovery\n```bash\necho 'MICRO_SWE_AGENT_FINAL_OUTPUT'\necho 'recovered from interrupt'\n```",
            ]
        ),
        env=LocalEnvironment(),
        problem_statement=problem_statement,
    )

    # Mock the query to raise KeyboardInterrupt on first call, then work normally
    original_query = agent.query
    call_count = 0

    def mock_query(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            raise KeyboardInterrupt()
        return original_query(*args, **kwargs)

    # Mock console.input based on the interruption_input parameter
    input_call_count = 0

    def mock_input(prompt):
        nonlocal input_call_count
        input_call_count += 1
        if input_call_count == 1:
            return interruption_input  # For the interruption handling
        return ""  # Confirm all subsequent actions

    with patch("microswea.agents.interactive.console.input", side_effect=mock_input):
        with patch.object(agent, "query", side_effect=mock_query):
            exit_status, result = agent.run()

    assert exit_status == "Submitted"
    assert result == "recovered from interrupt"
    # Check that the expected interruption message was added
    interrupt_messages = [msg for msg in agent.messages if expected_message_fragment in msg.get("content", "")]
    assert len(interrupt_messages) == 1

    return agent, interrupt_messages[0]


def test_interruption_handling_with_message():
    """Test that interruption with user message is handled properly."""
    agent, interrupt_message = _test_interruption_helper("User interrupted", "Interrupted by user")

    # Additional verification specific to this test
    assert "User interrupted" in interrupt_message["content"]


def test_interruption_handling_empty_message():
    """Test that interruption with empty input is handled properly."""
    _test_interruption_helper("", "Temporary interruption caught")


def test_multiple_confirmations_and_commands():
    """Test complex interaction with multiple confirmations and commands."""
    with patch(
        "microswea.agents.interactive.console.input",
        side_effect=[
            "reject first",  # Reject first action
            "/h",  # Show help for second action
            "/y",  # After help, enter yolo mode
            "",  # After yolo mode enabled, confirm (but yolo mode will skip future confirmations)
        ],
    ):
        agent = InteractiveAgent(
            model=DeterministicModel(
                outputs=[
                    "First action\n```bash\necho 'first'\n```",
                    "Second action\n```bash\necho 'MICRO_SWE_AGENT_FINAL_OUTPUT'\necho 'complex flow completed'\n```",
                ]
            ),
            env=LocalEnvironment(),
            problem_statement="Test complex interaction flow",
        )

        exit_status, result = agent.run()
        assert exit_status == "Submitted"
        assert result == "complex flow completed"
        assert agent.config.confirm_actions is False  # Should be in yolo mode
        assert agent.model.n_calls == 2


def test_non_whitelisted_action_requires_confirmation():
    """Test that non-whitelisted actions still require confirmation."""
    with patch("microswea.agents.interactive.console.input", return_value=""):
        agent = InteractiveAgent(
            model=DeterministicModel(
                outputs=["Non-whitelisted\n```bash\necho 'MICRO_SWE_AGENT_FINAL_OUTPUT'\necho 'confirmed'\n```"]
            ),
            env=LocalEnvironment(),
            problem_statement="Test non-whitelisted action",
            whitelist_actions=[r"ls.*"],  # Only ls commands whitelisted
        )

        exit_status, result = agent.run()
        assert exit_status == "Submitted"
        assert result == "confirmed"

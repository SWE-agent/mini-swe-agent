from unittest.mock import patch

from microsweagent.agents.interactive import InteractiveAgent
from microsweagent.environments.local import LocalEnvironment
from microsweagent.models.test_models import DeterministicModel


def test_successful_completion_with_confirmation():
    """Test agent completes successfully when user confirms all actions."""
    with patch("microsweagent.agents.interactive.console.input", side_effect=[""]):  # Confirm action with Enter
        agent = InteractiveAgent(
            model=DeterministicModel(
                outputs=["Finishing\n```bash\necho 'MICRO_SWE_AGENT_FINAL_OUTPUT'\necho 'completed'\n```"]
            ),
            env=LocalEnvironment(),
        )

        exit_status, result = agent.run("Test completion with confirmation")
        assert exit_status == "Submitted"
        assert result == "completed"
        assert agent.model.n_calls == 1


def test_action_rejection_and_recovery():
    """Test agent handles action rejection and can recover."""
    with patch(
        "microsweagent.agents.interactive.console.input",
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
        )

        exit_status, result = agent.run("Test action rejection")
        assert exit_status == "Submitted"
        assert result == "recovered"
        assert agent.model.n_calls == 2
        # Should have rejection message in conversation
        rejection_messages = [msg for msg in agent.messages if "User rejected this action" in msg.get("content", "")]
        assert len(rejection_messages) == 1


def test_yolo_mode_activation():
    """Test entering yolo mode disables confirmations."""
    with patch(
        "microsweagent.agents.interactive.console.input",
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
        )

        exit_status, result = agent.run("Test yolo mode")
        assert exit_status == "Submitted"
        assert result == "yolo works"
        assert agent.config.mode == "yolo"


def test_help_command():
    """Test help command shows help and continues normally."""
    with patch(
        "microsweagent.agents.interactive.console.input",
        side_effect=[
            "/h",  # Show help
            "",  # Confirm action after help
        ],
    ):
        with patch("microsweagent.agents.interactive.console.print") as mock_print:
            agent = InteractiveAgent(
                model=DeterministicModel(
                    outputs=["Test help\n```bash\necho 'MICRO_SWE_AGENT_FINAL_OUTPUT'\necho 'help shown'\n```"]
                ),
                env=LocalEnvironment(),
            )

            exit_status, result = agent.run("Test help command")
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
        whitelist_actions=[r"echo.*"],
    )

    # No patch needed - should not ask for confirmation
    exit_status, result = agent.run("Test whitelisted actions")
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

    with patch("microsweagent.agents.interactive.console.input", side_effect=mock_input):
        with patch.object(agent, "query", side_effect=mock_query):
            exit_status, result = agent.run(problem_statement)

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
        "microsweagent.agents.interactive.console.input",
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
        )

        exit_status, result = agent.run("Test complex interaction flow")
        assert exit_status == "Submitted"
        assert result == "complex flow completed"
        assert agent.config.mode == "yolo"  # Should be in yolo mode
        assert agent.model.n_calls == 2


def test_non_whitelisted_action_requires_confirmation():
    """Test that non-whitelisted actions still require confirmation."""
    with patch("microsweagent.agents.interactive.console.input", return_value=""):
        agent = InteractiveAgent(
            model=DeterministicModel(
                outputs=["Non-whitelisted\n```bash\necho 'MICRO_SWE_AGENT_FINAL_OUTPUT'\necho 'confirmed'\n```"]
            ),
            env=LocalEnvironment(),
            whitelist_actions=[r"ls.*"],  # Only ls commands whitelisted
        )

        exit_status, result = agent.run("Test non-whitelisted action")
        assert exit_status == "Submitted"
        assert result == "confirmed"


# New comprehensive mode switching tests


def test_human_mode_basic_functionality():
    """Test human mode where user enters shell commands directly."""
    with patch(
        "microsweagent.agents.interactive.console.input",
        side_effect=[
            "echo 'user command'",  # User enters shell command
            "echo 'MICRO_SWE_AGENT_FINAL_OUTPUT'\necho 'human mode works'",  # User enters final command
        ],
    ):
        agent = InteractiveAgent(
            model=DeterministicModel(outputs=[]),  # LM shouldn't be called in human mode
            env=LocalEnvironment(),
            mode="human",
        )

        exit_status, result = agent.run("Test human mode")
        assert exit_status == "Submitted"
        assert result == "human mode works"
        assert agent.config.mode == "human"
        assert agent.model.n_calls == 0  # LM should not be called


def test_human_mode_switch_to_yolo():
    """Test switching from human mode to yolo mode."""
    with patch(
        "microsweagent.agents.interactive.console.input",
        side_effect=[
            "/y",  # Switch to yolo mode from human mode
            "",  # Confirm action in yolo mode (though no confirmation needed)
        ],
    ):
        agent = InteractiveAgent(
            model=DeterministicModel(
                outputs=["LM action\n```bash\necho 'MICRO_SWE_AGENT_FINAL_OUTPUT'\necho 'switched to yolo'\n```"]
            ),
            env=LocalEnvironment(),
            mode="human",
        )

        exit_status, result = agent.run("Test human to yolo switch")
        assert exit_status == "Submitted"
        assert result == "switched to yolo"
        assert agent.config.mode == "yolo"
        assert agent.model.n_calls == 1


def test_human_mode_switch_to_confirm():
    """Test switching from human mode to confirm mode."""
    with patch(
        "microsweagent.agents.interactive.console.input",
        side_effect=[
            "/c",  # Switch to confirm mode from human mode
            "",  # Confirm action in confirm mode
        ],
    ):
        agent = InteractiveAgent(
            model=DeterministicModel(
                outputs=["LM action\n```bash\necho 'MICRO_SWE_AGENT_FINAL_OUTPUT'\necho 'switched to confirm'\n```"]
            ),
            env=LocalEnvironment(),
            mode="human",
        )

        exit_status, result = agent.run("Test human to confirm switch")
        assert exit_status == "Submitted"
        assert result == "switched to confirm"
        assert agent.config.mode == "confirm"
        assert agent.model.n_calls == 1


def test_confirmation_mode_switch_to_human_with_rejection():
    """Test switching from confirm mode to human mode with /u command."""
    with patch(
        "microsweagent.agents.interactive.console.input",
        side_effect=[
            "/u",  # Switch to human mode and reject action
            "echo 'MICRO_SWE_AGENT_FINAL_OUTPUT'\necho 'human command after rejection'",  # Human command
        ],
    ):
        agent = InteractiveAgent(
            model=DeterministicModel(
                outputs=[
                    "LM action\n```bash\necho 'first action'\n```",
                    "Recovery action\n```bash\necho 'recovery'\n```",
                ]
            ),
            env=LocalEnvironment(),
            mode="confirm",
        )

        exit_status, result = agent.run("Test confirm to human switch")
        assert exit_status == "Submitted"
        assert result == "human command after rejection"
        assert agent.config.mode == "human"
        # Should have rejection message
        rejection_messages = [msg for msg in agent.messages if "Switching to human mode" in msg.get("content", "")]
        assert len(rejection_messages) == 1


def test_confirmation_mode_switch_to_yolo_and_continue():
    """Test switching from confirm mode to yolo mode with /y and continuing with action."""
    with patch(
        "microsweagent.agents.interactive.console.input",
        side_effect=[
            "/y",  # Switch to yolo mode and confirm current action
        ],
    ):
        agent = InteractiveAgent(
            model=DeterministicModel(
                outputs=["LM action\n```bash\necho 'MICRO_SWE_AGENT_FINAL_OUTPUT'\necho 'switched and continued'\n```"]
            ),
            env=LocalEnvironment(),
            mode="confirm",
        )

        exit_status, result = agent.run("Test confirm to yolo switch")
        assert exit_status == "Submitted"
        assert result == "switched and continued"
        assert agent.config.mode == "yolo"


def test_mode_switch_during_keyboard_interrupt():
    """Test mode switching during keyboard interrupt handling."""
    agent = InteractiveAgent(
        model=DeterministicModel(
            outputs=[
                "Initial step\n```bash\necho 'will be interrupted'\n```",
                "Recovery\n```bash\necho 'MICRO_SWE_AGENT_FINAL_OUTPUT'\necho 'recovered after mode switch'\n```",
            ]
        ),
        env=LocalEnvironment(),
        mode="confirm",
    )

    # Mock the query to raise KeyboardInterrupt on first call
    original_query = agent.query
    call_count = 0

    def mock_query(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            raise KeyboardInterrupt()
        return original_query(*args, **kwargs)

    with patch(
        "microsweagent.agents.interactive.console.input",
        side_effect=[
            "/y",  # Switch to yolo mode during interrupt
            "",  # Confirm subsequent actions (though yolo mode won't ask)
        ],
    ):
        with patch.object(agent, "query", side_effect=mock_query):
            exit_status, result = agent.run("Test interrupt mode switch")

    assert exit_status == "Submitted"
    assert result == "recovered after mode switch"
    assert agent.config.mode == "yolo"
    # Should have interruption message
    interrupt_messages = [msg for msg in agent.messages if "Temporary interruption caught" in msg.get("content", "")]
    assert len(interrupt_messages) == 1


def test_already_in_mode_behavior():
    """Test behavior when trying to switch to the same mode."""
    with patch(
        "microsweagent.agents.interactive.console.input",
        side_effect=[
            "/c",  # Try to switch to confirm mode when already in confirm mode
            "",  # Confirm action after the "already in mode" recursive prompt
        ],
    ):
        agent = InteractiveAgent(
            model=DeterministicModel(
                outputs=["Test action\n```bash\necho 'MICRO_SWE_AGENT_FINAL_OUTPUT'\necho 'already in mode'\n```"]
            ),
            env=LocalEnvironment(),
            mode="confirm",
        )

        exit_status, result = agent.run("Test already in mode")
        assert exit_status == "Submitted"
        assert result == "already in mode"
        assert agent.config.mode == "confirm"


def test_all_mode_transitions_yolo_to_others():
    """Test transitions from yolo mode to other modes."""
    with patch(
        "microsweagent.agents.interactive.console.input",
        side_effect=[
            "/c",  # Switch from yolo to confirm
            "",  # Confirm action in confirm mode
        ],
    ):
        agent = InteractiveAgent(
            model=DeterministicModel(
                outputs=[
                    "First action\n```bash\necho 'yolo action'\n```",
                    "Second action\n```bash\necho 'MICRO_SWE_AGENT_FINAL_OUTPUT'\necho 'confirm action'\n```",
                ]
            ),
            env=LocalEnvironment(),
            mode="yolo",
        )

        # Trigger first action in yolo mode (should execute without confirmation)
        # Then interrupt to switch mode
        original_query = agent.query
        call_count = 0

        def mock_query(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 2:  # Interrupt on second query
                raise KeyboardInterrupt()
            return original_query(*args, **kwargs)

        with patch.object(agent, "query", side_effect=mock_query):
            exit_status, result = agent.run("Test yolo to confirm transition")

        assert exit_status == "Submitted"
        assert result == "confirm action"
        assert agent.config.mode == "confirm"


def test_all_mode_transitions_confirm_to_human():
    """Test transition from confirm mode to human mode."""
    with patch(
        "microsweagent.agents.interactive.console.input",
        side_effect=[
            "/u",  # Switch from confirm to human (rejecting action)
            "echo 'MICRO_SWE_AGENT_FINAL_OUTPUT'\necho 'human command'",  # User enters command in human mode
        ],
    ):
        agent = InteractiveAgent(
            model=DeterministicModel(outputs=["LM action\n```bash\necho 'rejected action'\n```"]),
            env=LocalEnvironment(),
            mode="confirm",
        )

        exit_status, result = agent.run("Test confirm to human transition")
        assert exit_status == "Submitted"
        assert result == "human command"
        assert agent.config.mode == "human"


def test_help_command_from_different_contexts():
    """Test help command works from different contexts (confirmation, interrupt, human mode)."""
    # Test help during confirmation
    with patch(
        "microsweagent.agents.interactive.console.input",
        side_effect=[
            "/h",  # Show help during confirmation
            "",  # Confirm after help
        ],
    ):
        with patch("microsweagent.agents.interactive.console.print") as mock_print:
            agent = InteractiveAgent(
                model=DeterministicModel(
                    outputs=["Test action\n```bash\necho 'MICRO_SWE_AGENT_FINAL_OUTPUT'\necho 'help works'\n```"]
                ),
                env=LocalEnvironment(),
                mode="confirm",
            )

            exit_status, result = agent.run("Test help from confirmation")
            assert exit_status == "Submitted"
            assert result == "help works"
            # Verify help was shown
            help_calls = [call for call in mock_print.call_args_list if "Current mode: " in str(call)]
            assert len(help_calls) > 0


def test_help_command_from_human_mode():
    """Test help command works from human mode."""
    with patch(
        "microsweagent.agents.interactive.console.input",
        side_effect=[
            "/h",  # Show help in human mode
            "echo 'MICRO_SWE_AGENT_FINAL_OUTPUT'\necho 'help in human mode'",  # User command after help
        ],
    ):
        with patch("microsweagent.agents.interactive.console.print") as mock_print:
            agent = InteractiveAgent(
                model=DeterministicModel(outputs=[]),  # LM shouldn't be called
                env=LocalEnvironment(),
                mode="human",
            )

            exit_status, result = agent.run("Test help from human mode")
            assert exit_status == "Submitted"
            assert result == "help in human mode"
            # Verify help was shown
            help_calls = [call for call in mock_print.call_args_list if "Current mode: " in str(call)]
            assert len(help_calls) > 0


def test_complex_mode_switching_sequence():
    """Test complex sequence of mode switches across different contexts."""
    agent = InteractiveAgent(
        model=DeterministicModel(
            outputs=[
                "Action 1\n```bash\necho 'action1'\n```",
                "Action 2\n```bash\necho 'action2'\n```",
                "Action 3\n```bash\necho 'MICRO_SWE_AGENT_FINAL_OUTPUT'\necho 'final action'\n```",
            ]
        ),
        env=LocalEnvironment(),
        mode="confirm",
    )

    # Mock interruption on second query
    original_query = agent.query
    call_count = 0

    def mock_query(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        if call_count == 2:
            raise KeyboardInterrupt()
        return original_query(*args, **kwargs)

    with patch(
        "microsweagent.agents.interactive.console.input",
        side_effect=[
            "/y",  # Confirm->Yolo during first action confirmation
            "/u",  # Yolo->Human during interrupt
            "/c",  # Human->Confirm in human mode
            "",  # Confirm final action
            "",  # Additional confirmation for any extra prompts
            "",  # Additional confirmation for any extra prompts
        ],
    ):
        with patch.object(agent, "query", side_effect=mock_query):
            exit_status, result = agent.run("Test complex mode switching")

    assert exit_status == "Submitted"
    assert result == "final action"
    assert agent.config.mode == "confirm"  # Should end in confirm mode

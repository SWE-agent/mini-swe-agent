from pathlib import Path
from unittest.mock import patch

import pytest
import yaml

from minisweagent.agents.interactive import InteractiveAgent
from minisweagent.environments.local import LocalEnvironment
from minisweagent.models.test_models import DeterministicModel


@pytest.fixture
def default_config():
    """Load default agent config from config/default.yaml"""
    config_path = Path("src/minisweagent/config/default.yaml")
    with open(config_path) as f:
        config = yaml.safe_load(f)
    return config["agent"]


def test_successful_completion_with_confirmation(default_config):
    """Test agent completes successfully when user confirms all actions."""
    with patch(
        "minisweagent.agents.interactive.prompt_session.prompt", side_effect=["", ""]
    ):  # Confirm action with Enter, then no new task
        agent = InteractiveAgent(
            model=DeterministicModel(
                outputs=["Finishing\n```bash\necho 'COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT'\necho 'completed'\n```"]
            ),
            env=LocalEnvironment(),
            **default_config,
        )

        exit_status, result = agent.run("Test completion with confirmation")
        assert exit_status == "Submitted"
        assert result == "completed\n"
        assert agent.model.n_calls == 1


def test_action_rejection_and_recovery(default_config):
    """Test agent handles action rejection and can recover."""
    with patch(
        "minisweagent.agents.interactive.prompt_session.prompt",
        side_effect=[
            "User rejected this action",  # Reject first action
            "",  # Confirm second action
            "",  # No new task when agent wants to finish
        ],
    ):
        agent = InteractiveAgent(
            model=DeterministicModel(
                outputs=[
                    "First try\n```bash\necho 'first attempt'\n```",
                    "Second try\n```bash\necho 'COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT'\necho 'recovered'\n```",
                ]
            ),
            env=LocalEnvironment(),
            **default_config,
        )

        exit_status, result = agent.run("Test action rejection")
        assert exit_status == "Submitted"
        assert result == "recovered\n"
        assert agent.model.n_calls == 2
        # Should have rejection message in conversation
        rejection_messages = [msg for msg in agent.messages if "User rejected this action" in msg.get("content", "")]
        assert len(rejection_messages) == 1


def test_yolo_mode_activation(default_config):
    """Test entering yolo mode disables confirmations."""
    with patch(
        "minisweagent.agents.interactive.prompt_session.prompt",
        side_effect=[
            "/y",  # Enter yolo mode
            "",  # This should be ignored since yolo mode is on
            "",  # No new task when agent wants to finish
        ],
    ):
        agent = InteractiveAgent(
            model=DeterministicModel(
                outputs=["Test command\n```bash\necho 'COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT'\necho 'yolo works'\n```"]
            ),
            env=LocalEnvironment(),
            **default_config,
        )

        exit_status, result = agent.run("Test yolo mode")
        assert exit_status == "Submitted"
        assert result == "yolo works\n"
        assert agent.config.mode == "yolo"


def test_help_command(default_config):
    """Test help command shows help and continues normally."""
    with patch(
        "minisweagent.agents.interactive.prompt_session.prompt",
        side_effect=[
            "/h",  # Show help
            "",  # Confirm action after help
            "",  # No new task when agent wants to finish
        ],
    ):
        with patch("minisweagent.agents.interactive.console.print") as mock_print:
            agent = InteractiveAgent(
                model=DeterministicModel(
                    outputs=["Test help\n```bash\necho 'COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT'\necho 'help shown'\n```"]
                ),
                env=LocalEnvironment(),
                **default_config,
            )

            exit_status, result = agent.run("Test help command")
            assert exit_status == "Submitted"
            assert result == "help shown\n"
            # Check that help was printed
            help_calls = [call for call in mock_print.call_args_list if "/y" in str(call)]
            assert len(help_calls) > 0


def test_whitelisted_actions_skip_confirmation(default_config):
    """Test that whitelisted actions don't require confirmation."""
    with patch(
        "minisweagent.agents.interactive.prompt_session.prompt",
        side_effect=[""],  # No new task when agent wants to finish
    ):
        agent = InteractiveAgent(
            model=DeterministicModel(
                outputs=[
                    "Whitelisted\n```bash\necho 'COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT'\necho 'no confirmation needed'\n```"
                ]
            ),
            env=LocalEnvironment(),
            **{
                **default_config,
                "whitelist_actions": [r"echo.*"],
            },
        )

        exit_status, result = agent.run("Test whitelisted actions")
        assert exit_status == "Submitted"
        assert result == "no confirmation needed\n"


def _test_interruption_helper(
    default_config, interruption_input, expected_message_fragment, problem_statement="Test interruption"
):
    """Helper function for testing interruption scenarios."""
    agent = InteractiveAgent(
        model=DeterministicModel(
            outputs=[
                "Initial step\n```bash\necho 'will be interrupted'\n```",
                "Recovery\n```bash\necho 'COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT'\necho 'recovered from interrupt'\n```",
            ]
        ),
        env=LocalEnvironment(),
        **default_config,
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

    with patch("minisweagent.agents.interactive.prompt_session.prompt", side_effect=mock_input):
        with patch.object(agent, "query", side_effect=mock_query):
            exit_status, result = agent.run(problem_statement)

    assert exit_status == "Submitted"
    assert result == "recovered from interrupt\n"
    # Check that the expected interruption message was added
    interrupt_messages = [msg for msg in agent.messages if expected_message_fragment in msg.get("content", "")]
    assert len(interrupt_messages) == 1

    return agent, interrupt_messages[0]


def test_interruption_handling_with_message(default_config):
    """Test that interruption with user message is handled properly."""
    agent, interrupt_message = _test_interruption_helper(default_config, "User interrupted", "Interrupted by user")

    # Additional verification specific to this test
    assert "User interrupted" in interrupt_message["content"]


def test_interruption_handling_empty_message(default_config):
    """Test that interruption with empty input is handled properly."""
    _test_interruption_helper(default_config, "", "Temporary interruption caught")


def test_multiple_confirmations_and_commands(default_config):
    """Test complex interaction with multiple confirmations and commands."""
    with patch(
        "minisweagent.agents.interactive.prompt_session.prompt",
        side_effect=[
            "reject first",  # Reject first action
            "/h",  # Show help for second action
            "/y",  # After help, enter yolo mode
            "",  # After yolo mode enabled, confirm (but yolo mode will skip future confirmations)
            "",  # No new task when agent wants to finish
        ],
    ):
        agent = InteractiveAgent(
            model=DeterministicModel(
                outputs=[
                    "First action\n```bash\necho 'first'\n```",
                    "Second action\n```bash\necho 'COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT'\necho 'complex flow completed'\n```",
                ]
            ),
            env=LocalEnvironment(),
            **default_config,
        )

        exit_status, result = agent.run("Test complex interaction flow")
        assert exit_status == "Submitted"
        assert result == "complex flow completed\n"
        assert agent.config.mode == "yolo"  # Should be in yolo mode
        assert agent.model.n_calls == 2


def test_non_whitelisted_action_requires_confirmation(default_config):
    """Test that non-whitelisted actions still require confirmation."""
    with patch(
        "minisweagent.agents.interactive.prompt_session.prompt",
        side_effect=["", ""],  # Confirm action, then no new task
    ):
        agent = InteractiveAgent(
            model=DeterministicModel(
                outputs=[
                    "Non-whitelisted\n```bash\necho 'COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT'\necho 'confirmed'\n```"
                ]
            ),
            env=LocalEnvironment(),
            **{
                **default_config,
                "whitelist_actions": [r"ls.*"],  # Only ls commands whitelisted
            },
        )

        exit_status, result = agent.run("Test non-whitelisted action")
        assert exit_status == "Submitted"
        assert result == "confirmed\n"


# New comprehensive mode switching tests


def test_human_mode_basic_functionality(default_config):
    """Test human mode where user enters shell commands directly."""
    with patch(
        "minisweagent.agents.interactive.prompt_session.prompt",
        side_effect=[
            "echo 'user command'",  # User enters shell command
            "echo 'COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT'\necho 'human mode works'",  # User enters final command
            "",  # No new task when agent wants to finish
        ],
    ):
        agent = InteractiveAgent(
            model=DeterministicModel(outputs=[]),  # LM shouldn't be called in human mode
            env=LocalEnvironment(),
            **{
                **default_config,
                "mode": "human",
            },
        )

        exit_status, result = agent.run("Test human mode")
        assert exit_status == "Submitted"
        assert result == "human mode works\n"
        assert agent.config.mode == "human"
        assert agent.model.n_calls == 0  # LM should not be called


def test_human_mode_switch_to_yolo(default_config):
    """Test switching from human mode to yolo mode."""
    with patch(
        "minisweagent.agents.interactive.prompt_session.prompt",
        side_effect=[
            "/y",  # Switch to yolo mode from human mode
            "",  # Confirm action in yolo mode (though no confirmation needed)
            "",  # No new task when agent wants to finish
        ],
    ):
        agent = InteractiveAgent(
            model=DeterministicModel(
                outputs=[
                    "LM action\n```bash\necho 'COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT'\necho 'switched to yolo'\n```"
                ]
            ),
            env=LocalEnvironment(),
            **{
                **default_config,
                "mode": "human",
            },
        )

        exit_status, result = agent.run("Test human to yolo switch")
        assert exit_status == "Submitted"
        assert result == "switched to yolo\n"
        assert agent.config.mode == "yolo"
        assert agent.model.n_calls == 1


def test_human_mode_switch_to_confirm(default_config):
    """Test switching from human mode to confirm mode."""
    with patch(
        "minisweagent.agents.interactive.prompt_session.prompt",
        side_effect=[
            "/c",  # Switch to confirm mode from human mode
            "",  # Confirm action in confirm mode
            "",  # No new task when agent wants to finish
        ],
    ):
        agent = InteractiveAgent(
            model=DeterministicModel(
                outputs=[
                    "LM action\n```bash\necho 'COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT'\necho 'switched to confirm'\n```"
                ]
            ),
            env=LocalEnvironment(),
            **{
                **default_config,
                "mode": "human",
            },
        )

        exit_status, result = agent.run("Test human to confirm switch")
        assert exit_status == "Submitted"
        assert result == "switched to confirm\n"
        assert agent.config.mode == "confirm"
        assert agent.model.n_calls == 1


def test_confirmation_mode_switch_to_human_with_rejection(default_config):
    """Test switching from confirm mode to human mode with /u command."""
    with patch(
        "minisweagent.agents.interactive.prompt_session.prompt",
        side_effect=[
            "/u",  # Switch to human mode and reject action
            "echo 'COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT'\necho 'human command after rejection'",  # Human command
            "",  # No new task when agent wants to finish
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
            **{
                **default_config,
                "mode": "confirm",
            },
        )

        exit_status, result = agent.run("Test confirm to human switch")
        assert exit_status == "Submitted"
        assert result == "human command after rejection\n"
        assert agent.config.mode == "human"
        # Should have rejection message
        rejection_messages = [msg for msg in agent.messages if "Switching to human mode" in msg.get("content", "")]
        assert len(rejection_messages) == 1


def test_confirmation_mode_switch_to_yolo_and_continue(default_config):
    """Test switching from confirm mode to yolo mode with /y and continuing with action."""
    with patch(
        "minisweagent.agents.interactive.prompt_session.prompt",
        side_effect=[
            "/y",  # Switch to yolo mode and confirm current action
            "",  # No new task when agent wants to finish
        ],
    ):
        agent = InteractiveAgent(
            model=DeterministicModel(
                outputs=[
                    "LM action\n```bash\necho 'COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT'\necho 'switched and continued'\n```"
                ]
            ),
            env=LocalEnvironment(),
            **{
                **default_config,
                "mode": "confirm",
            },
        )

        exit_status, result = agent.run("Test confirm to yolo switch")
        assert exit_status == "Submitted"
        assert result == "switched and continued\n"
        assert agent.config.mode == "yolo"


def test_mode_switch_during_keyboard_interrupt(default_config):
    """Test mode switching during keyboard interrupt handling."""
    agent = InteractiveAgent(
        model=DeterministicModel(
            outputs=[
                "Initial step\n```bash\necho 'will be interrupted'\n```",
                "Recovery\n```bash\necho 'COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT'\necho 'recovered after mode switch'\n```",
            ]
        ),
        env=LocalEnvironment(),
        **{
            **default_config,
            "mode": "confirm",
        },
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
        "minisweagent.agents.interactive.prompt_session.prompt",
        side_effect=[
            "/y",  # Switch to yolo mode during interrupt
            "",  # Confirm subsequent actions (though yolo mode won't ask)
        ],
    ):
        with patch.object(agent, "query", side_effect=mock_query):
            exit_status, result = agent.run("Test interrupt mode switch")

    assert exit_status == "Submitted"
    assert result == "recovered after mode switch\n"
    assert agent.config.mode == "yolo"
    # Should have interruption message
    interrupt_messages = [msg for msg in agent.messages if "Temporary interruption caught" in msg.get("content", "")]
    assert len(interrupt_messages) == 1


def test_already_in_mode_behavior(default_config):
    """Test behavior when trying to switch to the same mode."""
    with patch(
        "minisweagent.agents.interactive.prompt_session.prompt",
        side_effect=[
            "/c",  # Try to switch to confirm mode when already in confirm mode
            "",  # Confirm action after the "already in mode" recursive prompt
            "",  # No new task when agent wants to finish
        ],
    ):
        agent = InteractiveAgent(
            model=DeterministicModel(
                outputs=[
                    "Test action\n```bash\necho 'COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT'\necho 'already in mode'\n```"
                ]
            ),
            env=LocalEnvironment(),
            **{
                **default_config,
                "mode": "confirm",
            },
        )

        exit_status, result = agent.run("Test already in mode")
        assert exit_status == "Submitted"
        assert result == "already in mode\n"
        assert agent.config.mode == "confirm"


def test_all_mode_transitions_yolo_to_others(default_config):
    """Test transitions from yolo mode to other modes."""
    with patch(
        "minisweagent.agents.interactive.prompt_session.prompt",
        side_effect=[
            "/c",  # Switch from yolo to confirm
            "",  # Confirm action in confirm mode
            "",  # No new task when agent wants to finish
        ],
    ):
        agent = InteractiveAgent(
            model=DeterministicModel(
                outputs=[
                    "First action\n```bash\necho 'yolo action'\n```",
                    "Second action\n```bash\necho 'COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT'\necho 'confirm action'\n```",
                ]
            ),
            env=LocalEnvironment(),
            **{
                **default_config,
                "mode": "yolo",
            },
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
        assert result == "confirm action\n"
        assert agent.config.mode == "confirm"


def test_all_mode_transitions_confirm_to_human(default_config):
    """Test transition from confirm mode to human mode."""
    with patch(
        "minisweagent.agents.interactive.prompt_session.prompt",
        side_effect=[
            "/u",  # Switch from confirm to human (rejecting action)
            "echo 'COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT'\necho 'human command'",  # User enters command in human mode
            "",  # No new task when agent wants to finish
        ],
    ):
        agent = InteractiveAgent(
            model=DeterministicModel(outputs=["LM action\n```bash\necho 'rejected action'\n```"]),
            env=LocalEnvironment(),
            **{
                **default_config,
                "mode": "confirm",
            },
        )

        exit_status, result = agent.run("Test confirm to human transition")
        assert exit_status == "Submitted"
        assert result == "human command\n"
        assert agent.config.mode == "human"


def test_help_command_from_different_contexts(default_config):
    """Test help command works from different contexts (confirmation, interrupt, human mode)."""
    # Test help during confirmation
    with patch(
        "minisweagent.agents.interactive.prompt_session.prompt",
        side_effect=[
            "/h",  # Show help during confirmation
            "",  # Confirm after help
            "",  # No new task when agent wants to finish
        ],
    ):
        with patch("minisweagent.agents.interactive.console.print") as mock_print:
            agent = InteractiveAgent(
                model=DeterministicModel(
                    outputs=[
                        "Test action\n```bash\necho 'COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT'\necho 'help works'\n```"
                    ]
                ),
                env=LocalEnvironment(),
                **{
                    **default_config,
                    "mode": "confirm",
                },
            )

            exit_status, result = agent.run("Test help from confirmation")
            assert exit_status == "Submitted"
            assert result == "help works\n"
            # Verify help was shown
            help_calls = [call for call in mock_print.call_args_list if "Current mode: " in str(call)]
            assert len(help_calls) > 0


def test_help_command_from_human_mode(default_config):
    """Test help command works from human mode."""
    with patch(
        "minisweagent.agents.interactive.prompt_session.prompt",
        side_effect=[
            "/h",  # Show help in human mode
            "echo 'COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT'\necho 'help in human mode'",  # User command after help
            "",  # No new task when agent wants to finish
        ],
    ):
        with patch("minisweagent.agents.interactive.console.print") as mock_print:
            agent = InteractiveAgent(
                model=DeterministicModel(outputs=[]),  # LM shouldn't be called
                env=LocalEnvironment(),
                **{
                    **default_config,
                    "mode": "human",
                },
            )

            exit_status, result = agent.run("Test help from human mode")
            assert exit_status == "Submitted"
            assert result == "help in human mode\n"
            # Verify help was shown
            help_calls = [call for call in mock_print.call_args_list if "Current mode: " in str(call)]
            assert len(help_calls) > 0


def test_complex_mode_switching_sequence(default_config):
    """Test complex sequence of mode switches across different contexts."""
    agent = InteractiveAgent(
        model=DeterministicModel(
            outputs=[
                "Action 1\n```bash\necho 'action1'\n```",
                "Action 2\n```bash\necho 'action2'\n```",
                "Action 3\n```bash\necho 'COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT'\necho 'final action'\n```",
            ]
        ),
        env=LocalEnvironment(),
        **{
            **default_config,
            "mode": "confirm",
        },
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
        "minisweagent.agents.interactive.prompt_session.prompt",
        side_effect=[
            "/y",  # Confirm->Yolo during first action confirmation
            "/u",  # Yolo->Human during interrupt
            "/c",  # Human->Confirm in human mode
            "",  # Confirm final action
            "",  # No new task when agent wants to finish
            "",  # Extra empty input for any additional prompts
            "",  # Extra empty input for any additional prompts
        ],
    ):
        with patch.object(agent, "query", side_effect=mock_query):
            exit_status, result = agent.run("Test complex mode switching")

    assert exit_status == "Submitted"
    assert result == "final action\n"
    assert agent.config.mode == "confirm"  # Should end in confirm mode


def test_limits_exceeded_with_user_continuation(default_config):
    """Test that when limits are exceeded, user can provide new limits and execution continues."""
    # Create agent with very low limits that will be exceeded
    agent = InteractiveAgent(
        model=DeterministicModel(
            outputs=[
                "Step 1\n```bash\necho 'first step'\n```",
                "Step 2\n```bash\necho 'second step'\n```",
                "Final step\n```bash\necho 'COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT'\necho 'completed after limit increase'\n```",
            ],
            cost_per_call=0.6,  # Will exceed cost_limit=0.5 on first call
        ),
        env=LocalEnvironment(),
        **{
            **default_config,
            "step_limit": 10,  # High enough to not interfere initially
            "cost_limit": 0.5,  # Will be exceeded with first model call (cost=0.6),
            "mode": "yolo",  # Use yolo mode to avoid confirmation prompts,
        },
    )

    # Mock input() to provide new limits when prompted
    with patch("builtins.input", side_effect=["10", "5.0"]):  # New step_limit=10, cost_limit=5.0
        with patch("minisweagent.agents.interactive.prompt_session.prompt", side_effect=[""]):  # No new task
            with patch("minisweagent.agents.interactive.console.print"):  # Suppress console output
                exit_status, result = agent.run("Test limits exceeded with continuation")

    assert exit_status == "Submitted"
    assert result == "completed after limit increase\n"
    assert agent.model.n_calls == 3  # Should complete all 3 steps
    assert agent.config.step_limit == 10  # Should have updated step limit
    assert agent.config.cost_limit == 5.0  # Should have updated cost limit


def test_limits_exceeded_multiple_times_with_continuation(default_config):
    """Test that limits can be exceeded and updated multiple times."""
    agent = InteractiveAgent(
        model=DeterministicModel(
            outputs=[
                "Step 1\n```bash\necho 'step1'\n```",
                "Step 2\n```bash\necho 'step2'\n```",
                "Step 3\n```bash\necho 'step3'\n```",
                "Step 4\n```bash\necho 'step4'\n```",
                "Final\n```bash\necho 'COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT'\necho 'completed after multiple increases'\n```",
            ],
            cost_per_call=1.0,  # Standard cost per call
        ),
        env=LocalEnvironment(),
        **{
            **default_config,
            "step_limit": 1,  # Will be exceeded after first step
            "cost_limit": 100.0,  # High enough to not interfere,
            "mode": "yolo",
        },
    )

    # Mock input() to provide new limits multiple times
    # First limit increase: step_limit=2, then step_limit=10 when exceeded again
    with patch("builtins.input", side_effect=["2", "100.0", "10", "100.0"]):
        with patch("minisweagent.agents.interactive.prompt_session.prompt", side_effect=[""]):  # No new task
            with patch("minisweagent.agents.interactive.console.print"):
                exit_status, result = agent.run("Test multiple limit increases")

    assert exit_status == "Submitted"
    assert result == "completed after multiple increases\n"
    assert agent.model.n_calls == 5  # Should complete all 5 steps
    assert agent.config.step_limit == 10  # Should have final updated step limit


def test_continue_after_completion_with_new_task(default_config):
    """Test that user can provide a new task when agent wants to finish."""
    with patch(
        "minisweagent.agents.interactive.prompt_session.prompt",
        side_effect=[
            "",  # Confirm first action
            "Create a new file",  # Provide new task when agent wants to finish
            "",  # Confirm second action for new task
            "",  # Don't provide another task after second completion (finish)
        ],
    ):
        agent = InteractiveAgent(
            model=DeterministicModel(
                outputs=[
                    "First task\n```bash\necho 'COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT'\necho 'first task completed'\n```",
                    "Second task\n```bash\necho 'COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT'\necho 'new task completed'\n```",
                ]
            ),
            env=LocalEnvironment(),
            **default_config,
        )

        exit_status, result = agent.run("Complete the initial task")
        assert exit_status == "Submitted"
        assert result == "new task completed\n"
        assert agent.model.n_calls == 2
        # Should have the new task message in conversation
        new_task_messages = [
            msg for msg in agent.messages if "The user added a new task: Create a new file" in msg.get("content", "")
        ]
        assert len(new_task_messages) == 1


def test_continue_after_completion_without_new_task(default_config):
    """Test that agent finishes normally when user doesn't provide a new task."""
    with patch(
        "minisweagent.agents.interactive.prompt_session.prompt",
        side_effect=[
            "",  # Confirm first action
            "",  # Don't provide new task when agent wants to finish (empty input)
        ],
    ):
        agent = InteractiveAgent(
            model=DeterministicModel(
                outputs=[
                    "Task completion\n```bash\necho 'COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT'\necho 'original task completed'\n```"
                ]
            ),
            env=LocalEnvironment(),
            **default_config,
        )

        exit_status, result = agent.run("Complete the task")
        assert exit_status == "Submitted"
        assert result == "original task completed\n"
        assert agent.model.n_calls == 1
        # Should not have any new task messages
        new_task_messages = [msg for msg in agent.messages if "The user added a new task" in msg.get("content", "")]
        assert len(new_task_messages) == 0


def test_continue_after_completion_multiple_cycles(default_config):
    """Test multiple continuation cycles with new tasks."""
    with patch(
        "minisweagent.agents.interactive.prompt_session.prompt",
        side_effect=[
            "",  # Confirm first action
            "Second task",  # Provide first new task
            "",  # Confirm second action
            "Third task",  # Provide second new task
            "",  # Confirm third action
            "",  # Don't provide another task (finish)
        ],
    ):
        agent = InteractiveAgent(
            model=DeterministicModel(
                outputs=[
                    "First\n```bash\necho 'COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT'\necho 'first completed'\n```",
                    "Second\n```bash\necho 'COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT'\necho 'second completed'\n```",
                    "Third\n```bash\necho 'COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT'\necho 'third completed'\n```",
                ]
            ),
            env=LocalEnvironment(),
            **default_config,
        )

        exit_status, result = agent.run("Initial task")
        assert exit_status == "Submitted"
        assert result == "third completed\n"
        assert agent.model.n_calls == 3
        # Should have both new task messages
        new_task_messages = [msg for msg in agent.messages if "The user added a new task" in msg.get("content", "")]
        assert len(new_task_messages) == 2
        assert "Second task" in new_task_messages[0]["content"]
        assert "Third task" in new_task_messages[1]["content"]


def test_continue_after_completion_in_yolo_mode(default_config):
    """Test continuation when starting in yolo mode (no confirmations needed)."""
    with patch(
        "minisweagent.agents.interactive.prompt_session.prompt",
        side_effect=[
            "Create a second task",  # Provide new task when agent wants to finish
            "",  # Don't provide another task after second completion (finish)
        ],
    ):
        agent = InteractiveAgent(
            model=DeterministicModel(
                outputs=[
                    "First\n```bash\necho 'COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT'\necho 'first completed'\n```",
                    "Second\n```bash\necho 'COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT'\necho 'second task completed'\n```",
                ]
            ),
            env=LocalEnvironment(),
            **{
                **default_config,
                "mode": "yolo",  # Start in yolo mode
            },
        )

        exit_status, result = agent.run("Initial task")
        assert exit_status == "Submitted"
        assert result == "second task completed\n"
        assert agent.config.mode == "yolo"
        assert agent.model.n_calls == 2
        # Should have the new task message
        new_task_messages = [msg for msg in agent.messages if "Create a second task" in msg.get("content", "")]
        assert len(new_task_messages) == 1


def test_confirm_exit_enabled_asks_for_confirmation(default_config):
    """Test that when confirm_exit=True, agent asks for confirmation before finishing."""
    with patch(
        "minisweagent.agents.interactive.prompt_session.prompt",
        side_effect=["", ""],  # Confirm action, then no new task (empty string to exit)
    ):
        agent = InteractiveAgent(
            model=DeterministicModel(
                outputs=["Finishing\n```bash\necho 'COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT'\necho 'completed'\n```"]
            ),
            env=LocalEnvironment(),
            **{
                **default_config,
                "confirm_exit": True,  # Should ask for confirmation
            },
        )

        exit_status, result = agent.run("Test confirm exit enabled")
        assert exit_status == "Submitted"
        assert result == "completed\n"
        assert agent.model.n_calls == 1


def test_confirm_exit_disabled_exits_immediately(default_config):
    """Test that when confirm_exit=False, agent exits immediately without asking."""
    with patch(
        "minisweagent.agents.interactive.prompt_session.prompt",
        side_effect=[""],  # Only confirm action, no exit confirmation needed
    ):
        agent = InteractiveAgent(
            model=DeterministicModel(
                outputs=["Finishing\n```bash\necho 'COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT'\necho 'completed'\n```"]
            ),
            env=LocalEnvironment(),
            **{
                **default_config,
                "confirm_exit": False,  # Should NOT ask for confirmation
            },
        )

        exit_status, result = agent.run("Test confirm exit disabled")
        assert exit_status == "Submitted"
        assert result == "completed\n"
        assert agent.model.n_calls == 1


def test_confirm_exit_with_new_task_continues_execution(default_config):
    """Test that when user provides new task at exit confirmation, agent continues."""
    with patch(
        "minisweagent.agents.interactive.prompt_session.prompt",
        side_effect=[
            "",  # Confirm first action
            "Please do one more thing",  # Provide new task instead of exiting
            "",  # Confirm second action
            "",  # No new task on second exit confirmation
        ],
    ):
        agent = InteractiveAgent(
            model=DeterministicModel(
                outputs=[
                    "First task\n```bash\necho 'COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT'\necho 'first done'\n```",
                    "Additional task\n```bash\necho 'COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT'\necho 'additional done'\n```",
                ]
            ),
            env=LocalEnvironment(),
            **{
                **default_config,
                "confirm_exit": True,
            },
        )

        exit_status, result = agent.run("Test exit with new task")
        assert exit_status == "Submitted"
        assert result == "additional done\n"
        assert agent.model.n_calls == 2
        # Check that the new task was added to the conversation
        new_task_messages = [msg for msg in agent.messages if "Please do one more thing" in msg.get("content", "")]
        assert len(new_task_messages) == 1


def test_confirm_exit_config_field_defaults(default_config):
    """Test that confirm_exit field has correct default value."""
    agent = InteractiveAgent(
        model=DeterministicModel(outputs=[]),
        env=LocalEnvironment(),
        **default_config,
    )
    # Default should be True
    assert agent.config.confirm_exit is True


def test_confirm_exit_config_field_can_be_set(default_config):
    """Test that confirm_exit field can be explicitly set."""
    agent_with_confirm = InteractiveAgent(
        model=DeterministicModel(outputs=[]),
        env=LocalEnvironment(),
        **{
            **default_config,
            "confirm_exit": True,
        },
    )
    assert agent_with_confirm.config.confirm_exit is True

    agent_without_confirm = InteractiveAgent(
        model=DeterministicModel(outputs=[]),
        env=LocalEnvironment(),
        **{
            **default_config,
            "confirm_exit": False,
        },
    )
    assert agent_without_confirm.config.confirm_exit is False

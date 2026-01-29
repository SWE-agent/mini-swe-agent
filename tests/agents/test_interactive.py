from pathlib import Path
from unittest.mock import patch

import pytest
import yaml

from minisweagent.agents.interactive import InteractiveAgent
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


# --- Model factory functions ---


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


def _make_model(outputs: list[tuple[str, list[dict]]], **kwargs) -> DeterministicModel:
    """Create a DeterministicModel from a list of (content, actions) tuples.

    Kept for backward compatibility with tests that don't need parametrization.
    """
    return make_text_model(outputs, **kwargs)


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
    """Load toolcall agent config from config/mini_toolcall.yaml"""
    config_path = Path("src/minisweagent/config/mini_toolcall.yaml")
    with open(config_path) as f:
        config = yaml.safe_load(f)
    return config["agent"]


@pytest.fixture(params=["text", "toolcall", "response_api"])
def model_factory(request, default_config, toolcall_config):
    """Parametrized fixture that returns (factory_fn, config) for all three model types."""
    if request.param == "text":
        return make_text_model, default_config
    elif request.param == "toolcall":
        return make_tc_model, toolcall_config
    else:  # response_api
        return make_response_api_model, toolcall_config


def test_successful_completion_with_confirmation(model_factory):
    """Test agent completes successfully when user confirms all actions."""
    factory, config = model_factory
    with patch(
        "minisweagent.agents.interactive.prompt_session.prompt", side_effect=["", ""]
    ):  # Confirm action with Enter, then no new task
        agent = InteractiveAgent(
            model=factory(
                [
                    ("Finishing", [{"command": "echo 'COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT'\necho 'completed'"}]),
                ]
            ),
            env=LocalEnvironment(),
            **config,
        )

        info = agent.run("Test completion with confirmation")
        assert info["exit_status"] == "Submitted"
        assert info["submission"] == "completed\n"
        assert agent.n_calls == 1


def test_action_rejection_and_recovery(model_factory):
    """Test agent handles action rejection and can recover."""
    factory, config = model_factory
    with patch(
        "minisweagent.agents.interactive.prompt_session.prompt",
        side_effect=[
            "User rejected this action",  # Reject first action
            "",  # Confirm second action
            "",  # No new task when agent wants to finish
        ],
    ):
        agent = InteractiveAgent(
            model=factory(
                [
                    ("First try", [{"command": "echo 'first attempt'"}]),
                    ("Second try", [{"command": "echo 'COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT'\necho 'recovered'"}]),
                ]
            ),
            env=LocalEnvironment(),
            **config,
        )

        info = agent.run("Test action rejection")
        assert info["exit_status"] == "Submitted"
        assert info["submission"] == "recovered\n"
        assert agent.n_calls == 2
        # Should have rejection message in conversation
        rejection_messages = [msg for msg in agent.messages if "User rejected this action" in get_text(msg)]
        assert len(rejection_messages) == 1


def test_yolo_mode_activation(model_factory):
    """Test entering yolo mode disables confirmations."""
    factory, config = model_factory
    with patch(
        "minisweagent.agents.interactive.prompt_session.prompt",
        side_effect=[
            "/y",  # Enter yolo mode
            "",  # This should be ignored since yolo mode is on
            "",  # No new task when agent wants to finish
        ],
    ):
        agent = InteractiveAgent(
            model=factory(
                [
                    ("Test command", [{"command": "echo 'COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT'\necho 'yolo works'"}]),
                ]
            ),
            env=LocalEnvironment(),
            **config,
        )

        info = agent.run("Test yolo mode")
        assert info["exit_status"] == "Submitted"
        assert info["submission"] == "yolo works\n"
        assert agent.config.mode == "yolo"


def test_help_command(model_factory):
    """Test help command shows help and continues normally."""
    factory, config = model_factory
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
                model=factory(
                    [
                        ("Test help", [{"command": "echo 'COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT'\necho 'help shown'"}]),
                    ]
                ),
                env=LocalEnvironment(),
                **config,
            )

            info = agent.run("Test help command")
            assert info["exit_status"] == "Submitted"
            assert info["submission"] == "help shown\n"
            # Check that help was printed
            help_calls = [call for call in mock_print.call_args_list if "/y" in str(call)]
            assert len(help_calls) > 0


def test_whitelisted_actions_skip_confirmation(model_factory):
    """Test that whitelisted actions don't require confirmation."""
    factory, config = model_factory
    with patch(
        "minisweagent.agents.interactive.prompt_session.prompt",
        side_effect=[""],  # No new task when agent wants to finish
    ):
        agent = InteractiveAgent(
            model=factory(
                [
                    (
                        "Whitelisted",
                        [{"command": "echo 'COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT'\necho 'no confirmation needed'"}],
                    ),
                ]
            ),
            env=LocalEnvironment(),
            **{
                **config,
                "whitelist_actions": [r"echo.*"],
            },
        )

        info = agent.run("Test whitelisted actions")
        assert info["exit_status"] == "Submitted"
        assert info["submission"] == "no confirmation needed\n"


def _test_interruption_helper(
    factory, config, interruption_input, expected_message_fragment, problem_statement="Test interruption"
):
    """Helper function for testing interruption scenarios."""
    agent = InteractiveAgent(
        model=factory(
            [
                ("Initial step", [{"command": "echo 'will be interrupted'"}]),
                (
                    "Recovery",
                    [{"command": "echo 'COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT'\necho 'recovered from interrupt'"}],
                ),
            ]
        ),
        env=LocalEnvironment(),
        **config,
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
            info = agent.run(problem_statement)

    assert info["exit_status"] == "Submitted"
    assert info["submission"] == "recovered from interrupt\n"
    # Check that the expected interruption message was added
    interrupt_messages = [msg for msg in agent.messages if expected_message_fragment in get_text(msg)]
    assert len(interrupt_messages) == 1

    return agent, interrupt_messages[0]


def test_interruption_handling_with_message(model_factory):
    """Test that interruption with user message is handled properly."""
    factory, config = model_factory
    agent, interrupt_message = _test_interruption_helper(factory, config, "User interrupted", "Interrupted by user")

    # Additional verification specific to this test
    assert "User interrupted" in get_text(interrupt_message)


def test_interruption_handling_empty_message(model_factory):
    """Test that interruption with empty input is handled properly."""
    factory, config = model_factory
    _test_interruption_helper(factory, config, "", "Temporary interruption caught")


def test_multiple_confirmations_and_commands(model_factory):
    """Test complex interaction with multiple confirmations and commands."""
    factory, config = model_factory
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
            model=factory(
                [
                    ("First action", [{"command": "echo 'first'"}]),
                    (
                        "Second action",
                        [{"command": "echo 'COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT'\necho 'complex flow completed'"}],
                    ),
                ]
            ),
            env=LocalEnvironment(),
            **config,
        )

        info = agent.run("Test complex interaction flow")
        assert info["exit_status"] == "Submitted"
        assert info["submission"] == "complex flow completed\n"
        assert agent.config.mode == "yolo"  # Should be in yolo mode
        assert agent.n_calls == 2


def test_non_whitelisted_action_requires_confirmation(model_factory):
    """Test that non-whitelisted actions still require confirmation."""
    factory, config = model_factory
    with patch(
        "minisweagent.agents.interactive.prompt_session.prompt",
        side_effect=["", ""],  # Confirm action, then no new task
    ):
        agent = InteractiveAgent(
            model=factory(
                [
                    (
                        "Non-whitelisted",
                        [{"command": "echo 'COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT'\necho 'confirmed'"}],
                    ),
                ]
            ),
            env=LocalEnvironment(),
            **{
                **config,
                "whitelist_actions": [r"ls.*"],  # Only ls commands whitelisted
            },
        )

        info = agent.run("Test non-whitelisted action")
        assert info["exit_status"] == "Submitted"
        assert info["submission"] == "confirmed\n"


# New comprehensive mode switching tests


def test_human_mode_basic_functionality(model_factory):
    """Test human mode where user enters shell commands directly."""
    factory, config = model_factory
    with patch(
        "minisweagent.agents.interactive.prompt_session.prompt",
        side_effect=[
            "echo 'user command'",  # User enters shell command
            "echo 'COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT'\necho 'human mode works'",  # User enters final command
            "",  # No new task when agent wants to finish
        ],
    ):
        agent = InteractiveAgent(
            model=factory([]),  # LM shouldn't be called in human mode
            env=LocalEnvironment(),
            **{
                **config,
                "mode": "human",
            },
        )

        info = agent.run("Test human mode")
        assert info["exit_status"] == "Submitted"
        assert info["submission"] == "human mode works\n"
        assert agent.config.mode == "human"
        assert agent.n_calls == 0  # LM should not be called


def test_human_mode_switch_to_yolo(model_factory):
    """Test switching from human mode to yolo mode."""
    factory, config = model_factory
    with patch(
        "minisweagent.agents.interactive.prompt_session.prompt",
        side_effect=[
            "/y",  # Switch to yolo mode from human mode
            "",  # Confirm action in yolo mode (though no confirmation needed)
            "",  # No new task when agent wants to finish
        ],
    ):
        agent = InteractiveAgent(
            model=factory(
                [
                    (
                        "LM action",
                        [{"command": "echo 'COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT'\necho 'switched to yolo'"}],
                    ),
                ]
            ),
            env=LocalEnvironment(),
            **{
                **config,
                "mode": "human",
            },
        )

        info = agent.run("Test human to yolo switch")
        assert info["exit_status"] == "Submitted"
        assert info["submission"] == "switched to yolo\n"
        assert agent.config.mode == "yolo"
        assert agent.n_calls == 1


def test_human_mode_switch_to_confirm(model_factory):
    """Test switching from human mode to confirm mode."""
    factory, config = model_factory
    with patch(
        "minisweagent.agents.interactive.prompt_session.prompt",
        side_effect=[
            "/c",  # Switch to confirm mode from human mode
            "",  # Confirm action in confirm mode
            "",  # No new task when agent wants to finish
        ],
    ):
        agent = InteractiveAgent(
            model=factory(
                [
                    (
                        "LM action",
                        [{"command": "echo 'COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT'\necho 'switched to confirm'"}],
                    ),
                ]
            ),
            env=LocalEnvironment(),
            **{
                **config,
                "mode": "human",
            },
        )

        info = agent.run("Test human to confirm switch")
        assert info["exit_status"] == "Submitted"
        assert info["submission"] == "switched to confirm\n"
        assert agent.config.mode == "confirm"
        assert agent.n_calls == 1


def test_confirmation_mode_switch_to_human_with_rejection(model_factory):
    """Test switching from confirm mode to human mode with /u command."""
    factory, config = model_factory
    with patch(
        "minisweagent.agents.interactive.prompt_session.prompt",
        side_effect=[
            "/u",  # Switch to human mode and reject action
            "echo 'COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT'\necho 'human command after rejection'",  # Human command
            "",  # No new task when agent wants to finish
        ],
    ):
        agent = InteractiveAgent(
            model=factory(
                [
                    ("LM action", [{"command": "echo 'first action'"}]),
                    ("Recovery action", [{"command": "echo 'recovery'"}]),
                ]
            ),
            env=LocalEnvironment(),
            **{
                **config,
                "mode": "confirm",
            },
        )

        info = agent.run("Test confirm to human switch")
        assert info["exit_status"] == "Submitted"
        assert info["submission"] == "human command after rejection\n"
        assert agent.config.mode == "human"
        # Should have rejection message
        rejection_messages = [msg for msg in agent.messages if "Switching to human mode" in get_text(msg)]
        assert len(rejection_messages) == 1


def test_confirmation_mode_switch_to_yolo_and_continue(model_factory):
    """Test switching from confirm mode to yolo mode with /y and continuing with action."""
    factory, config = model_factory
    with patch(
        "minisweagent.agents.interactive.prompt_session.prompt",
        side_effect=[
            "/y",  # Switch to yolo mode and confirm current action
            "",  # No new task when agent wants to finish
        ],
    ):
        agent = InteractiveAgent(
            model=factory(
                [
                    (
                        "LM action",
                        [{"command": "echo 'COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT'\necho 'switched and continued'"}],
                    ),
                ]
            ),
            env=LocalEnvironment(),
            **{
                **config,
                "mode": "confirm",
            },
        )

        info = agent.run("Test confirm to yolo switch")
        assert info["exit_status"] == "Submitted"
        assert info["submission"] == "switched and continued\n"
        assert agent.config.mode == "yolo"


def test_mode_switch_during_keyboard_interrupt(model_factory):
    """Test mode switching during keyboard interrupt handling."""
    factory, config = model_factory
    agent = InteractiveAgent(
        model=factory(
            [
                ("Initial step", [{"command": "echo 'will be interrupted'"}]),
                (
                    "Recovery",
                    [{"command": "echo 'COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT'\necho 'recovered after mode switch'"}],
                ),
            ]
        ),
        env=LocalEnvironment(),
        **{
            **config,
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
            info = agent.run("Test interrupt mode switch")

    assert info["exit_status"] == "Submitted"
    assert info["submission"] == "recovered after mode switch\n"
    assert agent.config.mode == "yolo"
    # Should have interruption message
    interrupt_messages = [msg for msg in agent.messages if "Temporary interruption caught" in get_text(msg)]
    assert len(interrupt_messages) == 1


def test_already_in_mode_behavior(model_factory):
    """Test behavior when trying to switch to the same mode."""
    factory, config = model_factory
    with patch(
        "minisweagent.agents.interactive.prompt_session.prompt",
        side_effect=[
            "/c",  # Try to switch to confirm mode when already in confirm mode
            "",  # Confirm action after the "already in mode" recursive prompt
            "",  # No new task when agent wants to finish
        ],
    ):
        agent = InteractiveAgent(
            model=factory(
                [
                    (
                        "Test action",
                        [{"command": "echo 'COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT'\necho 'already in mode'"}],
                    ),
                ]
            ),
            env=LocalEnvironment(),
            **{
                **config,
                "mode": "confirm",
            },
        )

        info = agent.run("Test already in mode")
        assert info["exit_status"] == "Submitted"
        assert info["submission"] == "already in mode\n"
        assert agent.config.mode == "confirm"


def test_all_mode_transitions_yolo_to_others(model_factory):
    """Test transitions from yolo mode to other modes."""
    factory, config = model_factory
    with patch(
        "minisweagent.agents.interactive.prompt_session.prompt",
        side_effect=[
            "/c",  # Switch from yolo to confirm
            "",  # Confirm action in confirm mode
            "",  # No new task when agent wants to finish
        ],
    ):
        agent = InteractiveAgent(
            model=factory(
                [
                    ("First action", [{"command": "echo 'yolo action'"}]),
                    (
                        "Second action",
                        [{"command": "echo 'COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT'\necho 'confirm action'"}],
                    ),
                ]
            ),
            env=LocalEnvironment(),
            **{
                **config,
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
            info = agent.run("Test yolo to confirm transition")

        assert info["exit_status"] == "Submitted"
        assert info["submission"] == "confirm action\n"
        assert agent.config.mode == "confirm"


def test_all_mode_transitions_confirm_to_human(model_factory):
    """Test transition from confirm mode to human mode."""
    factory, config = model_factory
    with patch(
        "minisweagent.agents.interactive.prompt_session.prompt",
        side_effect=[
            "/u",  # Switch from confirm to human (rejecting action)
            "echo 'COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT'\necho 'human command'",  # User enters command in human mode
            "",  # No new task when agent wants to finish
        ],
    ):
        agent = InteractiveAgent(
            model=factory([("LM action", [{"command": "echo 'rejected action'"}])]),
            env=LocalEnvironment(),
            **{
                **config,
                "mode": "confirm",
            },
        )

        info = agent.run("Test confirm to human transition")
        assert info["exit_status"] == "Submitted"
        assert info["submission"] == "human command\n"
        assert agent.config.mode == "human"


def test_help_command_from_different_contexts(model_factory):
    """Test help command works from different contexts (confirmation, interrupt, human mode)."""
    factory, config = model_factory
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
                model=factory(
                    [
                        (
                            "Test action",
                            [{"command": "echo 'COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT'\necho 'help works'"}],
                        ),
                    ]
                ),
                env=LocalEnvironment(),
                **{
                    **config,
                    "mode": "confirm",
                },
            )

            info = agent.run("Test help from confirmation")
            assert info["exit_status"] == "Submitted"
            assert info["submission"] == "help works\n"
            # Verify help was shown
            help_calls = [call for call in mock_print.call_args_list if "Current mode: " in str(call)]
            assert len(help_calls) > 0


def test_help_command_from_human_mode(model_factory):
    """Test help command works from human mode."""
    factory, config = model_factory
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
                model=factory([]),  # LM shouldn't be called
                env=LocalEnvironment(),
                **{
                    **config,
                    "mode": "human",
                },
            )

            info = agent.run("Test help from human mode")
            assert info["exit_status"] == "Submitted"
            assert info["submission"] == "help in human mode\n"
            # Verify help was shown
            help_calls = [call for call in mock_print.call_args_list if "Current mode: " in str(call)]
            assert len(help_calls) > 0


def test_complex_mode_switching_sequence(model_factory):
    """Test complex sequence of mode switches across different contexts."""
    factory, config = model_factory
    agent = InteractiveAgent(
        model=factory(
            [
                ("Action 1", [{"command": "echo 'action1'"}]),
                ("Action 2", [{"command": "echo 'action2'"}]),
                ("Action 3", [{"command": "echo 'COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT'\necho 'final action'"}]),
            ]
        ),
        env=LocalEnvironment(),
        **{
            **config,
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
            info = agent.run("Test complex mode switching")

    assert info["exit_status"] == "Submitted"
    assert info["submission"] == "final action\n"
    assert agent.config.mode == "confirm"  # Should end in confirm mode


def test_limits_exceeded_with_user_continuation(model_factory):
    """Test that when limits are exceeded, user can provide new limits and execution continues."""
    factory, config = model_factory
    # Create agent with very low limits that will be exceeded
    agent = InteractiveAgent(
        model=factory(
            [
                ("Step 1", [{"command": "echo 'first step'"}]),
                ("Step 2", [{"command": "echo 'second step'"}]),
                (
                    "Final step",
                    [
                        {
                            "command": "echo 'COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT'\necho 'completed after limit increase'"
                        }
                    ],
                ),
            ],
            cost_per_call=0.6,  # Will exceed cost_limit=0.5 on first call
        ),
        env=LocalEnvironment(),
        **{
            **config,
            "step_limit": 10,  # High enough to not interfere initially
            "cost_limit": 0.5,  # Will be exceeded with first model call (cost=0.6),
            "mode": "yolo",  # Use yolo mode to avoid confirmation prompts,
        },
    )

    # Mock input() to provide new limits when prompted
    with patch("builtins.input", side_effect=["10", "5.0"]):  # New step_limit=10, cost_limit=5.0
        with patch("minisweagent.agents.interactive.prompt_session.prompt", side_effect=[""]):  # No new task
            with patch("minisweagent.agents.interactive.console.print"):  # Suppress console output
                info = agent.run("Test limits exceeded with continuation")

    assert info["exit_status"] == "Submitted"
    assert info["submission"] == "completed after limit increase\n"
    assert agent.n_calls == 3  # Should complete all 3 steps
    assert agent.config.step_limit == 10  # Should have updated step limit
    assert agent.config.cost_limit == 5.0  # Should have updated cost limit


def test_limits_exceeded_multiple_times_with_continuation(model_factory):
    """Test that limits can be exceeded and updated multiple times."""
    factory, config = model_factory
    agent = InteractiveAgent(
        model=factory(
            [
                ("Step 1", [{"command": "echo 'step1'"}]),
                ("Step 2", [{"command": "echo 'step2'"}]),
                ("Step 3", [{"command": "echo 'step3'"}]),
                ("Step 4", [{"command": "echo 'step4'"}]),
                (
                    "Final",
                    [
                        {
                            "command": "echo 'COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT'\necho 'completed after multiple increases'"
                        }
                    ],
                ),
            ],
            cost_per_call=1.0,  # Standard cost per call
        ),
        env=LocalEnvironment(),
        **{
            **config,
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
                info = agent.run("Test multiple limit increases")

    assert info["exit_status"] == "Submitted"
    assert info["submission"] == "completed after multiple increases\n"
    assert agent.n_calls == 5  # Should complete all 5 steps
    assert agent.config.step_limit == 10  # Should have final updated step limit


def test_continue_after_completion_with_new_task(model_factory):
    """Test that user can provide a new task when agent wants to finish."""
    factory, config = model_factory
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
            model=factory(
                [
                    (
                        "First task",
                        [{"command": "echo 'COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT'\necho 'first task completed'"}],
                    ),
                    (
                        "Second task",
                        [{"command": "echo 'COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT'\necho 'new task completed'"}],
                    ),
                ]
            ),
            env=LocalEnvironment(),
            **config,
        )

        info = agent.run("Complete the initial task")
        assert info["exit_status"] == "Submitted"
        assert info["submission"] == "new task completed\n"
        assert agent.n_calls == 2
        # Should have the new task message in conversation
        new_task_messages = [
            msg for msg in agent.messages if "The user added a new task: Create a new file" in get_text(msg)
        ]
        assert len(new_task_messages) == 1


def test_continue_after_completion_without_new_task(model_factory):
    """Test that agent finishes normally when user doesn't provide a new task."""
    factory, config = model_factory
    with patch(
        "minisweagent.agents.interactive.prompt_session.prompt",
        side_effect=[
            "",  # Confirm first action
            "",  # Don't provide new task when agent wants to finish (empty input)
        ],
    ):
        agent = InteractiveAgent(
            model=factory(
                [
                    (
                        "Task completion",
                        [{"command": "echo 'COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT'\necho 'original task completed'"}],
                    ),
                ]
            ),
            env=LocalEnvironment(),
            **config,
        )

        info = agent.run("Complete the task")
        assert info["exit_status"] == "Submitted"
        assert info["submission"] == "original task completed\n"
        assert agent.n_calls == 1
        # Should not have any new task messages
        new_task_messages = [msg for msg in agent.messages if "The user added a new task" in get_text(msg)]
        assert len(new_task_messages) == 0


def test_continue_after_completion_multiple_cycles(model_factory):
    """Test multiple continuation cycles with new tasks."""
    factory, config = model_factory
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
            model=factory(
                [
                    ("First", [{"command": "echo 'COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT'\necho 'first completed'"}]),
                    ("Second", [{"command": "echo 'COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT'\necho 'second completed'"}]),
                    ("Third", [{"command": "echo 'COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT'\necho 'third completed'"}]),
                ]
            ),
            env=LocalEnvironment(),
            **config,
        )

        info = agent.run("Initial task")
        assert info["exit_status"] == "Submitted"
        assert info["submission"] == "third completed\n"
        assert agent.n_calls == 3
        # Should have both new task messages
        new_task_messages = [msg for msg in agent.messages if "The user added a new task" in get_text(msg)]
        assert len(new_task_messages) == 2
        assert "Second task" in get_text(new_task_messages[0])
        assert "Third task" in get_text(new_task_messages[1])


def test_continue_after_completion_in_yolo_mode(model_factory):
    """Test continuation when starting in yolo mode (no confirmations needed)."""
    factory, config = model_factory
    with patch(
        "minisweagent.agents.interactive.prompt_session.prompt",
        side_effect=[
            "Create a second task",  # Provide new task when agent wants to finish
            "",  # Don't provide another task after second completion (finish)
        ],
    ):
        agent = InteractiveAgent(
            model=factory(
                [
                    ("First", [{"command": "echo 'COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT'\necho 'first completed'"}]),
                    (
                        "Second",
                        [{"command": "echo 'COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT'\necho 'second task completed'"}],
                    ),
                ]
            ),
            env=LocalEnvironment(),
            **{
                **config,
                "mode": "yolo",  # Start in yolo mode
            },
        )

        info = agent.run("Initial task")
        assert info["exit_status"] == "Submitted"
        assert info["submission"] == "second task completed\n"
        assert agent.config.mode == "yolo"
        assert agent.n_calls == 2
        # Should have the new task message
        new_task_messages = [msg for msg in agent.messages if "Create a second task" in get_text(msg)]
        assert len(new_task_messages) == 1


def test_confirm_exit_enabled_asks_for_confirmation(model_factory):
    """Test that when confirm_exit=True, agent asks for confirmation before finishing."""
    factory, config = model_factory
    with patch(
        "minisweagent.agents.interactive.prompt_session.prompt",
        side_effect=["", ""],  # Confirm action, then no new task (empty string to exit)
    ):
        agent = InteractiveAgent(
            model=factory(
                [
                    ("Finishing", [{"command": "echo 'COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT'\necho 'completed'"}]),
                ]
            ),
            env=LocalEnvironment(),
            **{
                **config,
                "confirm_exit": True,  # Should ask for confirmation
            },
        )

        info = agent.run("Test confirm exit enabled")
        assert info["exit_status"] == "Submitted"
        assert info["submission"] == "completed\n"
        assert agent.n_calls == 1


def test_confirm_exit_disabled_exits_immediately(model_factory):
    """Test that when confirm_exit=False, agent exits immediately without asking."""
    factory, config = model_factory
    with patch(
        "minisweagent.agents.interactive.prompt_session.prompt",
        side_effect=[""],  # Only confirm action, no exit confirmation needed
    ):
        agent = InteractiveAgent(
            model=factory(
                [
                    ("Finishing", [{"command": "echo 'COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT'\necho 'completed'"}]),
                ]
            ),
            env=LocalEnvironment(),
            **{
                **config,
                "confirm_exit": False,  # Should NOT ask for confirmation
            },
        )

        info = agent.run("Test confirm exit disabled")
        assert info["exit_status"] == "Submitted"
        assert info["submission"] == "completed\n"
        assert agent.n_calls == 1


def test_confirm_exit_with_new_task_continues_execution(model_factory):
    """Test that when user provides new task at exit confirmation, agent continues."""
    factory, config = model_factory
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
            model=factory(
                [
                    ("First task", [{"command": "echo 'COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT'\necho 'first done'"}]),
                    (
                        "Additional task",
                        [{"command": "echo 'COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT'\necho 'additional done'"}],
                    ),
                ]
            ),
            env=LocalEnvironment(),
            **{
                **config,
                "confirm_exit": True,
            },
        )

        info = agent.run("Test exit with new task")
        assert info["exit_status"] == "Submitted"
        assert info["submission"] == "additional done\n"
        assert agent.n_calls == 2
        # Check that the new task was added to the conversation
        new_task_messages = [msg for msg in agent.messages if "Please do one more thing" in get_text(msg)]
        assert len(new_task_messages) == 1


def test_confirm_exit_config_field_defaults(model_factory):
    """Test that confirm_exit field has correct default value."""
    factory, config = model_factory
    agent = InteractiveAgent(
        model=factory([]),
        env=LocalEnvironment(),
        **config,
    )
    # Default should be True
    assert agent.config.confirm_exit is True


def test_confirm_exit_config_field_can_be_set(model_factory):
    """Test that confirm_exit field can be explicitly set."""
    factory, config = model_factory
    agent_with_confirm = InteractiveAgent(
        model=factory([]),
        env=LocalEnvironment(),
        **{
            **config,
            "confirm_exit": True,
        },
    )
    assert agent_with_confirm.config.confirm_exit is True

    agent_without_confirm = InteractiveAgent(
        model=factory([]),
        env=LocalEnvironment(),
        **{
            **config,
            "confirm_exit": False,
        },
    )
    assert agent_without_confirm.config.confirm_exit is False

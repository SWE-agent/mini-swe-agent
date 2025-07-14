import logging
from unittest.mock import Mock

import pytest

from microsweagent.agents.interactive_textual import AddLogEmitCallback, AgentApp
from microsweagent.environments.local import LocalEnvironment
from microsweagent.models.test_models import DeterministicModel


def get_screen_text(app: AgentApp) -> str:
    """Extract all text content from the app's UI."""
    text_parts = []

    # Get all Static widgets in the main content container
    content_container = app.query_one("#content")
    for static_widget in content_container.query("Static"):
        if static_widget.display:
            if hasattr(static_widget, "renderable") and static_widget.renderable:  # type: ignore[attr-defined]
                text_parts.append(str(static_widget.renderable))  # type: ignore[attr-defined]

    # Also check the confirmation container if it's visible
    if app.confirmation_container.display:
        for static_widget in app.confirmation_container.query("Static"):
            if static_widget.display:
                if hasattr(static_widget, "renderable") and static_widget.renderable:  # type: ignore[attr-defined]
                    text_parts.append(str(static_widget.renderable))  # type: ignore[attr-defined]

    return "\n".join(text_parts)


@pytest.mark.slow
async def test_everything_integration_test():
    app = AgentApp(
        model=DeterministicModel(
            outputs=[
                "/sleep 0.5",
                "THOUGHTT 1\n ```bash\necho '1'\n```",
                "THOUGHTT 2\n ```bash\necho '2'\n```",
                "THOUGHTT 3\n ```bash\necho '3'\n```",
                "THOUGHTT 4\n ```bash\necho '4'\n```",
                "FINISHING\n ```bash\necho 'MICRO_SWE_AGENT_FINAL_OUTPUT'\n```",
            ],
        ),
        env=LocalEnvironment(),
        task="What's up?",
        mode="confirm",
        cost_limit=10.0,
    )
    async with app.run_test() as pilot:
        assert app.agent_state == "RUNNING"
        assert "You are a helpful assistant that can do anything." in get_screen_text(app)
        assert "press enter" not in get_screen_text(app).lower()
        assert "Step 1/1" in app.title
        await pilot.pause(0.7)
        assert "Step 2/2" in app.title
        assert app.agent_state == "AWAITING_CONFIRMATION"
        assert "AWAITING_CONFIRMATION" in app.title
        assert "echo '1'" in get_screen_text(app)
        assert (
            "press [bold]enter[/bold] to confirm action or [bold]backspace[/bold] to reject"
            in get_screen_text(app).lower()
        )
        # Navigate to page 1
        await pilot.press("h")
        assert "Step 1/2" in app.title
        assert "You are a helpful assistant that can do anything." in get_screen_text(app)
        assert "press enter" not in get_screen_text(app).lower()
        await pilot.press("h")
        # should remain on same page
        assert "Step 1/2" in app.title
        assert "You are a helpful assistant that can do anything." in get_screen_text(app)
        # Back to current latest page
        await pilot.press("l")
        assert "Step 2/2" in app.title
        # Confirm directly with enter
        await pilot.press("enter")
        assert "Step 3/3" in app.title
        assert "AWAITING_CONFIRMATION" in app.title
        assert "echo '2'" in get_screen_text(app)
        # Reject with message
        await pilot.press("backspace")
        assert "Step 3/3" in app.title
        assert "AWAITING_CONFIRMATION" in app.title
        assert "echo '2'" in get_screen_text(app)  # unchanged
        await pilot.press("ctrl+d")
        await pilot.pause(0.1)
        assert "Step 4/4" in app.title
        assert "echo '3'" in get_screen_text(app)
        # Enter yolo mode
        assert pilot.app.agent.config.mode == "confirm"  # type: ignore[attr-defined]
        await pilot.press("y")
        assert pilot.app.agent.config.mode == "yolo"  # type: ignore[attr-defined]
        await pilot.press("enter")  # still need to confirm once for step 3
        # next action will be executed automatically, so we see step 5 next
        await pilot.pause(0.2)
        assert "Step 6/6" in app.title
        assert "echo 'MICRO_SWE_AGENT_FINAL_OUTPUT'" in get_screen_text(app)
        # await pilot.pause(0.1)
        assert "STOPPED" in app.title
        assert "press enter" not in get_screen_text(app).lower()
        # More navigation
        await pilot.press("0")
        assert "Step 1/6" in app.title
        assert "You are a helpful assistant that can do anything." in get_screen_text(app)
        await pilot.press("$")
        assert "Step 6/6" in app.title
        assert "MICRO_SWE_AGENT_FINAL_OUTPUT" in get_screen_text(app)


def test_messages_to_steps_edge_cases():
    """Test the _messages_to_steps function with various edge cases."""
    from microsweagent.agents.interactive_textual import _messages_to_steps

    # Empty messages
    assert _messages_to_steps([]) == []

    # Single system message
    messages = [{"role": "system", "content": "Hello"}]
    assert _messages_to_steps(messages) == [messages]

    # User message ends a step
    messages = [
        {"role": "system", "content": "System"},
        {"role": "assistant", "content": "Assistant"},
        {"role": "user", "content": "User1"},
        {"role": "assistant", "content": "Assistant2"},
        {"role": "user", "content": "User2"},
    ]
    expected = [
        [
            {"role": "system", "content": "System"},
            {"role": "assistant", "content": "Assistant"},
            {"role": "user", "content": "User1"},
        ],
        [{"role": "assistant", "content": "Assistant2"}, {"role": "user", "content": "User2"}],
    ]
    assert _messages_to_steps(messages) == expected

    # No user messages (incomplete step)
    messages = [
        {"role": "system", "content": "System"},
        {"role": "assistant", "content": "Assistant"},
    ]
    expected = [messages]
    assert _messages_to_steps(messages) == expected


async def test_empty_agent_content():
    """Test app behavior with no messages."""
    app = AgentApp(
        model=DeterministicModel(outputs=[]),
        env=LocalEnvironment(),
        task="Empty test",
        mode="yolo",
    )
    async with app.run_test() as pilot:
        # Initially should show waiting message
        await pilot.pause(0.1)
        content = get_screen_text(app)
        assert "Waiting for agent to start" in content or "You are a helpful assistant" in content


async def test_log_message_filtering():
    """Test that warning and error log messages trigger notifications."""
    app = AgentApp(
        model=DeterministicModel(
            outputs=[
                "/warning Test warning message",
                "Normal response",
                "end: \n```\necho MICRO_SWE_AGENT_FINAL_OUTPUT\n```",
            ]
        ),
        env=LocalEnvironment(),
        task="Log test",
        mode="yolo",
    )

    # Mock the notify method to capture calls
    app.notify = Mock()

    async with app.run_test() as pilot:
        await pilot.pause(0.2)

        # Verify warning was emitted and handled (note the extra space in the actual format)
        app.notify.assert_any_call("[WARNING]  Test warning message", severity="warning")


async def test_list_content_rendering():
    """Test rendering of messages with list content vs string content."""
    # Create a model that will add messages with list content
    app = AgentApp(
        model=DeterministicModel(outputs=["Simple response\n```bash\necho 'MICRO_SWE_AGENT_FINAL_OUTPUT'\n```"]),
        env=LocalEnvironment(),
        task="Content test",
        mode="yolo",
    )

    async with app.run_test() as pilot:
        # Wait for the agent to finish its normal operation
        await pilot.pause(0.2)

        # Now manually add a message with list content to test rendering
        app.agent.messages.append({"role": "assistant", "content": [{"text": "Line 1"}, {"text": "Line 2"}]})

        # Trigger the message update logic to refresh step count and navigate to last step
        app.on_message_added()

        # Navigate to the last step to see our new message
        app.action_last_step()

        assert "Line 1\nLine 2" in get_screen_text(app)


async def test_confirmation_rejection_with_message():
    """Test rejecting an action with a custom message."""
    app = AgentApp(
        model=DeterministicModel(outputs=["Test thought\n```bash\necho 'test'\n```"]),
        env=LocalEnvironment(),
        task="Rejection test",
        mode="confirm",
    )

    async with app.run_test() as pilot:
        await pilot.pause(0.1)

        # Wait for confirmation prompt
        while app.agent_state != "AWAITING_CONFIRMATION":
            await pilot.pause(0.1)

        # Start rejection
        await pilot.press("backspace")
        assert app.confirmation_container.rejecting is True

        # Type rejection message
        rejection_input = app.confirmation_container.query_one("#rejection-input")
        rejection_input.text = "Not safe to run"  # type: ignore[attr-defined]

        # Submit rejection
        await pilot.press("ctrl+d")
        await pilot.pause(0.1)

        # Verify the command was rejected with the message
        assert "Command not executed: Not safe to run" in get_screen_text(app)


async def test_agent_with_cost_limit():
    """Test agent behavior when cost limit is exceeded."""
    app = AgentApp(
        model=DeterministicModel(outputs=["Response 1", "Response 2"]),
        env=LocalEnvironment(),
        task="Cost limit test",
        mode="yolo",
        cost_limit=0.01,  # Very low limit
    )

    # Set model cost to exceed limit
    app.agent.model.cost = 0.02

    # Mock the notify method to capture calls
    app.notify = Mock()

    async with app.run_test() as pilot:
        await pilot.pause(0.2)

        # Should eventually stop due to cost limit and notify with the exit status
        assert app.agent_state == "STOPPED"
        app.notify.assert_called_with("Agent finished with status: LimitsExceeded")


async def test_agent_with_step_limit():
    """Test agent behavior when step limit is exceeded."""
    app = AgentApp(
        model=DeterministicModel(outputs=["Response 1", "Response 2", "Response 3"]),
        env=LocalEnvironment(),
        task="Step limit test",
        mode="yolo",
        step_limit=2,
    )

    # Mock the notify method to capture calls
    app.notify = Mock()

    async with app.run_test() as pilot:
        await pilot.pause(0.3)

        # Should stop due to step limit and notify with the exit status
        assert app.agent_state == "STOPPED"
        app.notify.assert_called_with("Agent finished with status: LimitsExceeded")


async def test_agent_successful_completion_notification():
    """Test that agent completion with 'Submitted' status triggers notification."""
    app = AgentApp(
        model=DeterministicModel(
            outputs=["Completing task\n```bash\necho 'MICRO_SWE_AGENT_FINAL_OUTPUT'\necho 'success'\n```"]
        ),
        env=LocalEnvironment(),
        task="Completion test",
        mode="yolo",
    )

    # Mock the notify method to capture calls
    app.notify = Mock()

    async with app.run_test() as pilot:
        await pilot.pause(0.2)

        # Should finish with Submitted status and notify about completion
        assert app.agent_state == "STOPPED"
        app.notify.assert_any_call("Agent finished with status: Submitted")


async def test_whitelist_actions_bypass_confirmation():
    """Test that whitelisted actions bypass confirmation."""
    app = AgentApp(
        model=DeterministicModel(
            outputs=["Whitelisted action\n```bash\necho 'safe' && echo 'MICRO_SWE_AGENT_FINAL_OUTPUT'\n```"]
        ),
        env=LocalEnvironment(),
        task="Whitelist test",
        mode="confirm",
        whitelist_actions=[r"echo.*"],
    )

    async with app.run_test() as pilot:
        await pilot.pause(0.2)

        # Should execute without confirmation because echo is whitelisted
        assert app.agent_state != "AWAITING_CONFIRMATION"
        assert "echo 'safe'" in get_screen_text(app)


async def test_confirmation_container_multiple_actions():
    """Test confirmation container handling multiple actions in sequence."""
    app = AgentApp(
        model=DeterministicModel(
            outputs=[
                "First action\n```bash\necho '1'\n```",
                "Second action\n```bash\necho '2' && echo 'MICRO_SWE_AGENT_FINAL_OUTPUT'\n```",
            ]
        ),
        env=LocalEnvironment(),
        task="Multiple actions test",
        mode="confirm",
    )

    async with app.run_test() as pilot:
        await pilot.pause(0.1)

        # Confirm first action
        while app.agent_state != "AWAITING_CONFIRMATION":
            await pilot.pause(0.1)
        assert "echo '1'" in get_screen_text(app)
        await pilot.press("enter")

        # Wait for and confirm second action
        await pilot.pause(0.1)
        while app.agent_state != "AWAITING_CONFIRMATION":
            await pilot.pause(0.1)
        assert "echo '2'" in get_screen_text(app)
        await pilot.press("enter")


async def test_scrolling_behavior():
    """Test scrolling up and down behavior."""
    app = AgentApp(
        model=DeterministicModel(
            outputs=["Long response" * 100 + "\n```bash\necho 'MICRO_SWE_AGENT_FINAL_OUTPUT'\n```"]
        ),
        env=LocalEnvironment(),
        task="Scroll test",
        mode="yolo",
    )

    async with app.run_test() as pilot:
        await pilot.pause(0.1)

        # Test scrolling
        vs = app.query_one("VerticalScroll")
        initial_y = vs.scroll_target_y
        await pilot.press("j")  # scroll down
        assert vs.scroll_target_y > initial_y
        await pilot.press("k")  # scroll up


def test_log_handler_cleanup():
    """Test that log handler is properly cleaned up."""
    initial_handlers = len(logging.getLogger().handlers)

    app = AgentApp(
        model=DeterministicModel(outputs=["Simple response\n```bash\necho 'MICRO_SWE_AGENT_FINAL_OUTPUT'\n```"]),
        env=LocalEnvironment(),
        task="Cleanup test",
        mode="yolo",
    )

    # Handler should be added
    assert len(logging.getLogger().handlers) == initial_handlers + 1

    # Simulate unmount
    app.on_unmount()

    # Handler should be removed
    assert len(logging.getLogger().handlers) == initial_handlers


def test_add_log_emit_callback():
    """Test the AddLogEmitCallback handler directly."""

    callback_called = False
    test_record = None

    def test_callback(record):
        nonlocal callback_called, test_record
        callback_called = True
        test_record = record

    handler = AddLogEmitCallback(test_callback)

    # Create a log record
    record = logging.LogRecord(
        name="test", level=logging.WARNING, pathname="test.py", lineno=1, msg="Test message", args=(), exc_info=None
    )

    handler.emit(record)

    assert callback_called
    assert test_record == record

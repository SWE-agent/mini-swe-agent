import asyncio
import logging
import threading
from unittest.mock import Mock

import pytest

from minisweagent.agents.interactive_textual import AddLogEmitCallback, SmartInputContainer, TextualAgent
from minisweagent.environments.local import LocalEnvironment
from minisweagent.models.test_models import DeterministicModel


def get_screen_text(app: TextualAgent) -> str:
    """Extract all text content from the app's UI."""
    text_parts = [app.title]

    def _append_visible_static_text(container):
        for static_widget in container.query("Static"):
            if static_widget.display:
                if hasattr(static_widget, "content") and static_widget.content:  # type: ignore[attr-defined]
                    text_parts.append(str(static_widget.content))  # type: ignore[attr-defined]
                elif hasattr(static_widget, "renderable") and static_widget.renderable:  # type: ignore[attr-defined]
                    text_parts.append(str(static_widget.renderable))  # type: ignore[attr-defined]

    # Get all Static widgets in the main content container
    content_container = app.query_one("#content")
    _append_visible_static_text(content_container)

    # Also check the input container if it's visible
    if app.input_container.display:
        _append_visible_static_text(app.input_container)

    return "\n".join(text_parts)


async def type_text(pilot, text: str):
    """Type text character by character using pilot.press() to simulate real user input.

    This properly tests focus behavior and input handling, unlike setting .value directly.
    """
    for char in text:
        # Handle special characters that need key names instead of character literals
        if char == " ":
            await pilot.press("space")
        elif char == "\n":
            await pilot.press("enter")
        elif char == "\t":
            await pilot.press("tab")
        else:
            # For regular characters, pilot.press() can handle them directly
            await pilot.press(char)


@pytest.mark.slow
async def test_everything_integration_test():
    app = TextualAgent(
        model=DeterministicModel(
            outputs=[
                "/sleep 0.5",
                "THOUGHTT 1\n ```bash\necho '1'\n```",  # step 2
                "THOUGHTT 2\n ```bash\necho '2'\n```",  # step 3
                "THOUGHTT 3\n ```bash\necho '3'\n```",  # step 4
                "THOUGHTT 4\n ```bash\necho '4'\n```",  # step 5
                "THOUGHTT 5\n ```bash\necho '5'\n```",  # step 6
                "THOUGHTT 6\n ```bash\necho '6'\n```",  # step 7
                "FINISHING\n ```bash\necho 'COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT'\n```",
                "FINISHING2\n ```bash\necho 'COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT'\n```",
            ],
        ),
        env=LocalEnvironment(),
        mode="confirm",
        cost_limit=10.0,
    )
    assert app.agent.config.confirm_exit
    async with app.run_test() as pilot:
        # Start the agent with the task
        threading.Thread(target=lambda: app.agent.run("What's up?"), daemon=True).start()
        await pilot.pause(0.2)
        assert app.agent_state == "RUNNING"
        assert "You are a helpful assistant that can do anything." in get_screen_text(app)
        assert "press enter" not in get_screen_text(app).lower()
        assert "Step 1/1" in app.title

        print(">>> Agent autoforwards -> step 2, then waiting for input")
        await pilot.pause(0.7)
        assert "Step 2/2" in app.title
        assert app.agent_state == "AWAITING_INPUT"
        assert "AWAITING_INPUT" in app.title
        assert "echo '1'" in get_screen_text(app)
        assert "press enter to confirm or provide rejection reason" in get_screen_text(app).lower()

        print(">>> Confirm directly with enter first and we move on to page 3")
        print(get_screen_text(app))
        await pilot.press("enter")
        await pilot.pause(0.5)
        print("---")
        print(get_screen_text(app))
        print("--- if we didn't follow, here's some cluses")
        print(f"{pilot.app.i_step=}, {pilot.app.n_steps=}, {pilot.app._vscroll.scroll_target_y=}")  # type: ignore
        assert "Step 3/3" in app.title

        print(">>> Now, let's navigate to page 1")
        await pilot.press("escape")  # unfocus from the confirmation input
        await pilot.press("h")  # --> 2/3
        await pilot.press("h")
        assert "Step 1/3" in app.title
        assert "You are a helpful assistant that can do anything." in get_screen_text(app)
        assert "press enter" not in get_screen_text(app).lower()
        await pilot.press("h")
        # should remain on same page
        assert "Step 1/3" in app.title
        assert "You are a helpful assistant that can do anything." in get_screen_text(app)

        print(">>> Back to current latest page, because we're stilling waiting for confirmation")
        await pilot.press("l")  # no need for escape, because confirmation is only on last page
        assert "Step 2/3" in app.title
        await pilot.press("l")  # no need for escape, because confirmation is only on last page
        assert "Step 3/3" in app.title
        assert "AWAITING_INPUT" in app.title
        assert "echo '2'" in get_screen_text(app)

        print(">>> Reject with message - type rejection reason and submit")
        await type_text(pilot, "Not safe to execute")
        await pilot.press("enter")
        print(get_screen_text(app))
        await pilot.pause(0.3)
        assert "Step 4/4" in app.title
        assert "echo '3'" in get_screen_text(app)

        print(">>> Reject with message multiline input")
        await pilot.press("ctrl+t")
        await type_text(pilot, "Not safe to execute\n")
        await pilot.press("ctrl+d")
        print(get_screen_text(app))
        await pilot.pause(0.3)
        assert "Step 5/5" in app.title
        assert "echo '4'" in get_screen_text(app)

        print(">>> Switch tohuman mode")
        await pilot.press("escape")
        await pilot.press("u")

        assert pilot.app.agent.config.mode == "human"  # type: ignore[attr-defined]
        await pilot.pause(0.2)
        print(get_screen_text(app))
        assert "User switched to manual mode, this command will be ignored" in get_screen_text(app)
        assert "Enter your command" in get_screen_text(app)
        assert "Step 5/5" in app.title  # we didn't move because waiting for human command

        print(">>> Human gives command")
        await type_text(pilot, "echo 'human'")
        await pilot.press("enter")
        await pilot.pause(0.2)
        print(get_screen_text(app))
        assert "Step 6/6" in app.title
        assert "human" in get_screen_text(app)  # show the observation

        print(">>> Enter yolo mode & confirm")
        await pilot.press("escape")
        assert pilot.app.agent.config.mode == "human"  # type: ignore[attr-defined]
        await pilot.press("y")
        # Note that this will add one step, because we're basically now executing an empty human action
        assert pilot.app.agent.config.mode == "yolo"  # type: ignore[attr-defined]
        # await pilot.press("enter")  # still need to confirm once for step 3
        # next action will be executed automatically, so we see step 6 next
        await pilot.pause(0.2)
        assert "Step 10/10" in app.title
        assert "echo 'COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT'" in get_screen_text(app)
        # await pilot.pause(0.1)
        # assert "press enter" not in get_screen_text(app).lower()
        print(get_screen_text(app))
        assert "AWAITING_INPUT" in app.title  # still waiting for confirmation of exit

        print(">>> Directly navigate to step 1")
        await pilot.press("escape")
        await pilot.press("0")
        assert "Step 1/10" in app.title
        assert "You are a helpful assistant that can do anything." in get_screen_text(app)

        print(">>> Directly navigate to step 9")
        await pilot.press("$")
        assert "Step 10/10" in app.title
        assert "COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT" in get_screen_text(app)

        print(">>> Give it a new task")
        assert "to give it a new task" in get_screen_text(app).lower()
        await type_text(pilot, "New task")
        await pilot.press("enter")
        await pilot.pause(0.2)

        print(">>> Exit confirmation should appear again")
        assert "Step 11/11" in app.title
        # assert "New task" in get_screen_text(app)
        assert "to give it a new task" in get_screen_text(app).lower()
        await pilot.press("enter")


def test_messages_to_steps_edge_cases():
    """Test the _messages_to_steps function with various edge cases."""
    from minisweagent.agents.interactive_textual import _messages_to_steps

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
    app = TextualAgent(
        model=DeterministicModel(outputs=[]),
        env=LocalEnvironment(),
        mode="yolo",
    )
    async with app.run_test() as pilot:
        # Start the agent with the task
        threading.Thread(target=lambda: app.agent.run("Empty test"), daemon=True).start()
        # Initially should show waiting message
        await pilot.pause(0.1)
        content = get_screen_text(app)
        assert "Waiting for agent to start" in content or "You are a helpful assistant" in content


async def test_log_message_filtering():
    """Test that warning and error log messages trigger notifications."""
    app = TextualAgent(
        model=DeterministicModel(
            outputs=[
                "/warning Test warning message",
                "Normal response",
                "end: \n```bash\necho COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT\n```",
            ]
        ),
        env=LocalEnvironment(),
        mode="yolo",
    )

    # Mock the notify method to capture calls
    app.notify = Mock()

    async with app.run_test() as pilot:
        # Wait for UI to be ready before starting the agent thread
        await pilot.pause(0.2)

        # Start the agent with the task
        threading.Thread(target=lambda: app.agent.run("Log test"), daemon=True).start()

        # Wait for notify to be called (with timeout)
        for _ in range(50):  # 5 sec timeout
            await pilot.pause(0.1)
            if app.notify.call_count > 0:
                break

        # Verify warning was emitted and handled (note the extra space in the actual format)
        app.notify.assert_any_call("[WARNING]  Test warning message", severity="warning")


async def test_list_content_rendering():
    """Test rendering of messages with list content vs string content."""
    # Create a model that will add messages with list content
    app = TextualAgent(
        model=DeterministicModel(
            outputs=["Simple response\n```bash\necho 'COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT'\n```"]
        ),
        env=LocalEnvironment(),
        mode="yolo",
    )

    async with app.run_test() as pilot:
        # Start the agent with the task
        threading.Thread(target=lambda: app.agent.run("Content test"), daemon=True).start()
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
    app = TextualAgent(
        model=DeterministicModel(outputs=["Test thought\n```bash\necho 'test'\n```"]),
        env=LocalEnvironment(),
        mode="confirm",
    )

    async with app.run_test() as pilot:
        # Start the agent with the task
        threading.Thread(target=lambda: app.agent.run("Rejection test"), daemon=True).start()
        await pilot.pause(0.1)

        # Wait for input prompt
        while app.agent_state != "AWAITING_INPUT":
            await pilot.pause(0.1)

        # Type rejection message and submit
        await type_text(pilot, "Not safe to run")
        await pilot.press("enter")
        await pilot.pause(0.1)

        # Verify the command was rejected with the message
        assert "Command not executed: Not safe to run" in get_screen_text(app)


async def test_agent_with_cost_limit():
    """Test agent behavior when cost limit is exceeded."""
    app = TextualAgent(
        model=DeterministicModel(outputs=["Response 1", "Response 2"]),
        env=LocalEnvironment(),
        mode="yolo",
        cost_limit=0.01,  # Very low limit
    )

    app.notify = Mock()

    async with app.run_test() as pilot:
        threading.Thread(target=lambda: app.agent.run("Cost limit test"), daemon=True).start()
        for _ in range(50):
            await pilot.pause(0.1)
            if app.agent_state == "STOPPED":
                break
        else:
            raise AssertionError("Agent did not stop within 5 seconds")

        # Should eventually stop due to cost limit and notify with the exit status
        assert app.agent_state == "STOPPED"
        app.notify.assert_called_with("Agent finished with status: LimitsExceeded")


async def test_agent_with_step_limit():
    """Test agent behavior when step limit is exceeded."""
    app = TextualAgent(
        model=DeterministicModel(outputs=["Response 1", "Response 2", "Response 3"]),
        env=LocalEnvironment(),
        mode="yolo",
        step_limit=2,
    )

    app.notify = Mock()
    async with app.run_test() as pilot:
        # Start the agent with the task
        threading.Thread(target=lambda: app.agent.run("Step limit test"), daemon=True).start()
        for _ in range(50):
            await pilot.pause(0.1)
            if app.agent_state == "STOPPED":
                break
        else:
            raise AssertionError("Agent did not stop within 5 seconds")
        assert app.agent_state == "STOPPED"
        app.notify.assert_called_with("Agent finished with status: LimitsExceeded")


async def test_whitelist_actions_bypass_confirmation():
    """Test that whitelisted actions bypass confirmation."""
    app = TextualAgent(
        model=DeterministicModel(
            outputs=["Whitelisted action\n```bash\necho 'safe' && echo 'COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT'\n```"]
        ),
        env=LocalEnvironment(),
        mode="confirm",
        whitelist_actions=[r"echo.*"],
    )

    async with app.run_test() as pilot:
        await pilot.pause(0.2)  # Wait for UI to be ready
        threading.Thread(target=lambda: app.agent.run("Whitelist test"), daemon=True).start()

        for _ in range(10):
            await pilot.pause(0.1)
            if "echo 'safe'" in get_screen_text(app):
                break

        # Should execute without confirmation because echo is whitelisted
        assert app.agent_state != "AWAITING_INPUT"
        assert "echo 'safe'" in get_screen_text(app)


async def test_input_container_multiple_actions():
    """Test input container handling multiple actions in sequence."""
    app = TextualAgent(
        model=DeterministicModel(
            outputs=[
                "First action\n```bash\necho '1'\n```",
                "Second action\n```bash\necho '2' && echo 'COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT'\n```",
            ]
        ),
        env=LocalEnvironment(),
        mode="confirm",
    )

    async with app.run_test() as pilot:
        # Start the agent with the task
        threading.Thread(target=lambda: app.agent.run("Multiple actions test"), daemon=True).start()
        await pilot.pause(0.1)

        # Confirm first action
        while app.agent_state != "AWAITING_INPUT":
            await pilot.pause(0.1)
        assert "echo '1'" in get_screen_text(app)
        await pilot.press("enter")

        # Wait for and confirm second action
        await pilot.pause(0.1)
        while app.agent_state != "AWAITING_INPUT":
            await pilot.pause(0.1)
        assert "echo '2'" in get_screen_text(app)
        await pilot.press("enter")


def test_log_handler_cleanup():
    """Test that log handler is properly cleaned up."""
    initial_handlers = len(logging.getLogger().handlers)

    app = TextualAgent(
        model=DeterministicModel(
            outputs=["Simple response\n```bash\necho 'COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT'\n```"]
        ),
        env=LocalEnvironment(),
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


async def test_yolo_mode_confirms_pending_action():
    """Test that pressing 'y' to switch to YOLO mode also confirms any pending action."""
    app = TextualAgent(
        model=DeterministicModel(
            outputs=[
                "Action requiring confirmation\n```bash\necho 'test' && echo 'COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT'\n```",
            ]
        ),
        env=LocalEnvironment(),
        mode="confirm",
    )

    async with app.run_test() as pilot:
        # Start the agent with the task
        threading.Thread(target=lambda: app.agent.run("YOLO confirmation test"), daemon=True).start()
        await pilot.pause(0.1)

        # Wait for input prompt
        while app.agent_state != "AWAITING_INPUT":
            await pilot.pause(0.1)

        # Verify we're in confirm mode and awaiting input
        assert app.agent.config.mode == "confirm"
        assert app.agent_state == "AWAITING_INPUT"
        assert "echo 'test'" in get_screen_text(app)
        assert "press enter to confirm or provide rejection reason" in get_screen_text(app).lower()

        # Press 'y' to switch to YOLO mode - first escape from input focus
        await pilot.press("escape")
        await pilot.press("y")
        await pilot.pause(0.1)

        # Verify mode changed to yolo
        assert app.agent.config.mode == "yolo"

        # Since we escaped from input first, we still need to confirm the action
        # Navigate back to the input step and confirm
        await pilot.press("$")  # Go to last step
        if app.agent_state == "AWAITING_INPUT":
            await pilot.press("enter")  # Confirm the action
            await pilot.pause(0.1)


# ===== SmartInputContainer Unit Tests =====

from textual.app import App
from textual.containers import Container


class DummyTestApp(App):
    """Minimal test app for providing Textual context."""

    def __init__(self):
        super().__init__()
        self.agent_state = "RUNNING"
        self.call_from_thread = Mock()
        self.update_content = Mock()
        self.set_focus = Mock()
        self._vscroll = Mock()
        self._vscroll.scroll_y = 0

    def compose(self):
        yield Container()


def create_mock_smart_input_container(app):
    """Create SmartInputContainer with proper mocking to avoid type issues."""
    from typing import cast

    # Create actual SmartInputContainer instance but use typing.cast to bypass type check
    return SmartInputContainer(cast("TextualAgent", app))  # type: ignore


async def test_smart_input_container_initialization():
    """Test SmartInputContainer initialization and default state."""
    app = DummyTestApp()
    async with app.run_test():
        container = create_mock_smart_input_container(app)

        assert container._app == app
        assert container._multiline_mode is False
        assert container.can_focus is True
        assert container.display is False
        assert container.pending_prompt is None
        assert container._input_result is None
        assert isinstance(container._input_event, threading.Event)


async def test_smart_input_container_request_input():
    """Test request_input method behavior."""
    app = DummyTestApp()
    async with app.run_test():
        container = create_mock_smart_input_container(app)

        # Start request_input in a thread since it blocks
        test_prompt = "Test prompt"
        result_container = {}

        def request_thread():
            result_container["result"] = container.request_input(test_prompt)

        thread = threading.Thread(target=request_thread)
        thread.start()

        # Give thread time to start and set up
        await asyncio.sleep(0.1)

        # Check that prompt was set
        assert container.pending_prompt == test_prompt
        assert app.call_from_thread.called

        # Complete the input with empty string (confirmation)
        container._complete_input("")

        # Wait for thread to complete
        thread.join(timeout=1)

        # Check results
        assert result_container["result"] == ""
        assert container.pending_prompt is None
        assert container.display is False


async def test_smart_input_container_complete_input():
    """Test _complete_input method resets state correctly."""
    app = DummyTestApp()
    async with app.run_test():
        container = SmartInputContainer(app)

        # Set up initial state
        container.pending_prompt = "Test prompt"
        container._multiline_mode = True
        container.display = True
        container._single_input.value = "test"
        container._multi_input.text = "test\nmultiline"

        # Complete input
        test_result = "User input result"
        container._complete_input(test_result)

        # Check state reset
        assert container._input_result == test_result
        assert container.pending_prompt is None
        assert container.display is False
        assert container._single_input.value == ""
        assert container._multi_input.text == ""
        assert container._multiline_mode is False
        assert app.agent_state == "RUNNING"
        assert app.update_content.called
        assert app._vscroll.scroll_y == 0


async def test_smart_input_container_toggle_mode():
    """Test switching from single-line to multi-line mode."""
    app = DummyTestApp()
    async with app.run_test():
        container = SmartInputContainer(app)

        # Set up single-line mode with content
        container.pending_prompt = "Test prompt"
        container._multiline_mode = False
        container._single_input.value = "test input"

        # Mock focus method
        container.on_focus = Mock()

        # Toggle to multiline mode
        container.action_toggle_mode()

        # Check mode changed
        assert container._multiline_mode is True
        assert container.on_focus.called


async def test_smart_input_container_toggle_mode_blocked():
    """Test that toggle mode is blocked when no pending prompt or already in multiline."""
    app = DummyTestApp()
    async with app.run_test():
        container = SmartInputContainer(app)

        # Test with no pending prompt
        container.pending_prompt = None
        initial_mode = container._multiline_mode
        container.action_toggle_mode()
        assert container._multiline_mode == initial_mode

        # Test when already in multiline mode
        container.pending_prompt = "Test prompt"
        container._multiline_mode = True
        container.action_toggle_mode()
        assert container._multiline_mode is True


async def test_smart_input_container_single_input_submission():
    """Test single-line input submission."""
    app = DummyTestApp()
    async with app.run_test():
        container = SmartInputContainer(app)

        # Set up for single-line mode
        container._multiline_mode = False
        container.pending_prompt = "Test prompt"

        # Mock the complete_input method
        container._complete_input = Mock()

        # Create mock input event
        mock_input = Mock()
        mock_input.id = "single-input"
        mock_input.value = "  test input  "

        mock_event = Mock()
        mock_event.input = mock_input

        # Trigger submission
        container.on_input_submitted(mock_event)

        # Check that complete_input was called with stripped text
        container._complete_input.assert_called_once_with("test input")


async def test_smart_input_container_single_input_submission_multiline_mode():
    """Test that single-line submission is ignored in multiline mode."""
    app = DummyTestApp()
    async with app.run_test():
        container = SmartInputContainer(app)

        # Set up for multiline mode
        container._multiline_mode = True
        container._complete_input = Mock()

        # Create mock input event
        mock_input = Mock()
        mock_input.id = "single-input"
        mock_input.value = "test input"

        mock_event = Mock()
        mock_event.input = mock_input

        # Trigger submission
        container.on_input_submitted(mock_event)

        # Check that complete_input was NOT called
        container._complete_input.assert_not_called()


async def test_smart_input_container_key_events():
    """Test key event handling for various scenarios."""
    app = DummyTestApp()
    async with app.run_test():
        container = SmartInputContainer(app)

        # Mock methods
        container.action_toggle_mode = Mock()
        container._complete_input = Mock()

        # Test Ctrl+T in single-line mode
        container._multiline_mode = False
        mock_event = Mock()
        mock_event.key = "ctrl+t"
        mock_event.prevent_default = Mock()

        container.on_key(mock_event)

        assert mock_event.prevent_default.called
        assert container.action_toggle_mode.called

        # Reset mocks
        container.action_toggle_mode.reset_mock()
        mock_event.prevent_default.reset_mock()

        # Test Ctrl+D in multiline mode
        container._multiline_mode = True
        container._multi_input.text = "  multiline\ntext  "
        mock_event.key = "ctrl+d"

        container.on_key(mock_event)

        assert mock_event.prevent_default.called
        container._complete_input.assert_called_once_with("multiline\ntext")

        # Reset mocks
        container._complete_input.reset_mock()
        mock_event.prevent_default.reset_mock()

        # Test Escape key
        mock_event.key = "escape"

        container.on_key(mock_event)

        assert mock_event.prevent_default.called
        assert container.can_focus is False
        app.set_focus.assert_called_once_with(None)


async def test_smart_input_container_key_events_no_action():
    """Test key events that should not trigger any action."""
    app = DummyTestApp()
    async with app.run_test():
        container = SmartInputContainer(app)

        # Mock methods
        container.action_toggle_mode = Mock()
        container._complete_input = Mock()

        # Test Ctrl+T in multiline mode (should not toggle)
        container._multiline_mode = True
        mock_event = Mock()
        mock_event.key = "ctrl+t"
        mock_event.prevent_default = Mock()

        container.on_key(mock_event)

        # Should not prevent default or call toggle
        assert not mock_event.prevent_default.called
        assert not container.action_toggle_mode.called

        # Test Ctrl+D in single-line mode (should not complete)
        container._multiline_mode = False
        mock_event.key = "ctrl+d"
        mock_event.prevent_default.reset_mock()

        container.on_key(mock_event)

        # Should not prevent default or complete input
        assert not mock_event.prevent_default.called
        assert not container._complete_input.called


async def test_smart_input_container_on_focus():
    """Test focus behavior in different modes."""
    app = DummyTestApp()
    async with app.run_test():
        container = SmartInputContainer(app)

        # Mock focus methods
        container._single_input.focus = Mock()
        container._multi_input.focus = Mock()

        # Test focus in single-line mode
        container._multiline_mode = False
        container.on_focus()

        assert container._single_input.focus.called
        assert not container._multi_input.focus.called

        # Reset and test focus in multiline mode
        container._single_input.focus.reset_mock()
        container._multi_input.focus.reset_mock()
        container._multiline_mode = True
        container.on_focus()

        assert not container._single_input.focus.called
        assert container._multi_input.focus.called


async def test_smart_input_container_on_mount():
    """Test widget initialization on mount."""
    app = DummyTestApp()
    async with app.run_test():
        container = SmartInputContainer(app)

        # Mock update method
        container._update_mode_display = Mock()

        # Trigger mount
        container.on_mount()

        # Check initialization
        assert container._multi_input.display is False
        assert container._update_mode_display.called


async def test_system_commands_are_callable():
    """Test that all system commands returned by get_system_commands are callable.

    This prevents TypeError when commands are selected from the command palette,
    which requires callable methods instead of action name strings.
    """
    app = TextualAgent(
        model=DeterministicModel(outputs=["Test\n```bash\necho 'test'\n```"]),
        env=LocalEnvironment(),
        mode="yolo",
    )

    async with app.run_test():
        screen = app.screen
        commands = list(app.get_system_commands(screen))

        for command in commands:
            assert callable(command.callback), (
                f"Command '{command.title}' has non-callable callback: {command.callback}"
            )

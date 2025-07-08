from microswea.agents.interactive_textual import AgentApp
from microswea.environments.local import LocalEnvironment
from microswea.models.test_models import DeterministicModel


def get_screen_text(app: AgentApp) -> str:
    """Extract all text content from the app's UI."""
    text_parts = []

    # Get all Static widgets in the main content container
    content_container = app.query_one("#content")
    for static_widget in content_container.query("Static"):
        if static_widget.display:
            if hasattr(static_widget, "renderable") and static_widget.renderable:
                text_parts.append(str(static_widget.renderable))

    # Also check the confirmation container if it's visible
    if app.confirmation_container.display:
        for static_widget in app.confirmation_container.query("Static"):
            if static_widget.display:
                if hasattr(static_widget, "renderable") and static_widget.renderable:
                    text_parts.append(str(static_widget.renderable))

    return "\n".join(text_parts)


async def test_everything_integration_test():
    app = AgentApp(
        model=DeterministicModel(
            outputs=[
                "/sleep 0.3",
                "THOUGHTT 1\n ```bash\necho '1'\n```",
                "THOUGHTT 2\n ```bash\necho '2'\n```",
                "THOUGHTT 3\n ```bash\necho '3'\n```",
                "THOUGHTT 4\n ```bash\necho '4'\n```",
                "FINISHING\n ```bash\necho 'MICRO_SWE_AGENT_FINAL_OUTPUT'\n```",
            ],
        ),
        env=LocalEnvironment(),
        problem_statement="What's up?",
        confirm_actions=True,
    )
    async with app.run_test() as pilot:
        assert app.agent_state == "RUNNING"
        assert "You are a helpful assistant that can do anything." in get_screen_text(app)
        assert "press enter" not in get_screen_text(app).lower()
        assert "Step 1/1" in app.sub_title
        await pilot.pause(0.5)
        assert "Step 2/2" in app.sub_title
        assert app.agent_state == "AWAITING_CONFIRMATION"
        assert "AWAITING_CONFIRMATION" in app.sub_title
        assert "echo '1'" in get_screen_text(app)
        assert "press enter to confirm action or backspace to reject" in get_screen_text(app).lower()
        # Navigate to page 1
        await pilot.press("h")
        assert "Step 1/2" in app.sub_title
        assert "You are a helpful assistant that can do anything." in get_screen_text(app)
        assert "press enter" not in get_screen_text(app).lower()
        await pilot.press("h")
        # should remain on same page
        assert "Step 1/2" in app.sub_title
        assert "You are a helpful assistant that can do anything." in get_screen_text(app)
        # Back to current latest page
        await pilot.press("l")
        assert "Step 2/2" in app.sub_title
        # Confirm directly with enter
        await pilot.press("enter")
        assert "Step 3/3" in app.sub_title
        assert "AWAITING_CONFIRMATION" in app.sub_title
        assert "echo '2'" in get_screen_text(app)
        # Reject with message
        await pilot.press("backspace")
        assert "Step 3/3" in app.sub_title
        assert "AWAITING_CONFIRMATION" in app.sub_title
        assert "echo '2'" in get_screen_text(app)  # unchanged
        await pilot.press("ctrl+d")
        await pilot.pause(0.1)
        assert "Step 4/4" in app.sub_title
        assert "echo '3'" in get_screen_text(app)
        # Enter yolo mode
        assert pilot.app.agent.config.confirm_actions is True
        await pilot.press("y")
        assert pilot.app.agent.config.confirm_actions is False
        await pilot.press("enter")  # still need to confirm once for step 3
        # next action will be executed automatically, so we see step 5 next
        await pilot.pause(0.1)
        assert "Step 6/6" in app.sub_title
        assert "echo 'MICRO_SWE_AGENT_FINAL_OUTPUT'" in get_screen_text(app)
        # await pilot.pause(0.1)
        assert "STOPPED" in app.sub_title
        assert "press enter" not in get_screen_text(app).lower()
        # More navigation
        await pilot.press("0")
        assert "Step 1/6" in app.sub_title
        assert "You are a helpful assistant that can do anything." in get_screen_text(app)
        await pilot.press("$")
        assert "Step 6/6" in app.sub_title
        assert "MICRO_SWE_AGENT_FINAL_OUTPUT" in get_screen_text(app)

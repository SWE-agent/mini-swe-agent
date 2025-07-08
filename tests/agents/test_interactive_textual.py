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
                "/sleep 0.1",
                "THOUGHTT 1\n ```bash\necho '1'\n```",
                "THOUGHTT 2\n ```bash\necho '2'\n```",
                "THOUGHTT 3\n ```bash\necho '3'\n```",
                "THOUGHTT 4\n ```bash\necho '4'\n```",
            ],
        ),
        env=LocalEnvironment(),
        problem_statement="What's up?",
        confirm_actions=True,
    )
    async with app.run_test() as pilot:
        app.console.record = True
        # assert app.agent_state == "RUNNING"
        # assert "You are a helpful assistant that can do anything." in get_screen_text(app)
        await pilot.pause(0.2)
        # assert app.agent_state == "RUNNING"
        assert "AWAITING_CONFIRMATION" in app.sub_title
        assert "echo '1'" in get_screen_text(app)
        assert "press enter to confirm action or backspace to reject" in get_screen_text(app).lower()
        await pilot.press("enter")
        assert "AWAITING_CONFIRMATION" in app.sub_title
        assert "echo '2'" in get_screen_text(app)
        await pilot.press("backspace")
        assert "AWAITING_CONFIRMATION" in app.sub_title
        await pilot.press("ctrl+d")
        await pilot.pause(0.1)
        assert "echo '3'" in get_screen_text(app)
        # await pilot.pause(0.1)
        # assert "STOPPED" in app.sub_title

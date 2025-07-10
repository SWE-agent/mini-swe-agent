import json
import tempfile
from pathlib import Path

import pytest

from microswea.run.inspector import TrajectoryInspector


def get_screen_text(app: TrajectoryInspector) -> str:
    """Extract all text content from the app's UI."""
    text_parts = []

    # Get all Static widgets in the main content container
    content_container = app.query_one("#content")
    for static_widget in content_container.query("Static"):
        if static_widget.display:
            if hasattr(static_widget, "renderable") and static_widget.renderable:  # type: ignore[attr-defined]
                text_parts.append(str(static_widget.renderable))  # type: ignore[attr-defined]

    return "\n".join(text_parts)


@pytest.fixture
def sample_simple_trajectory():
    """Sample trajectory in simple format (list of messages)."""
    return [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Hello, solve this problem."},
        {"role": "assistant", "content": "I'll help you solve this.\n\n```bash\nls -la\n```"},
        {"role": "user", "content": "Command output here."},
        {"role": "assistant", "content": "Now I'll finish.\n\n```bash\necho MICRO_SWE_AGENT_FINAL_OUTPUT\n```"},
    ]


@pytest.fixture
def sample_swebench_trajectory():
    """Sample trajectory in SWEBench format (dict with messages array)."""
    return {
        "instance_id": "test-instance-1",
        "info": {
            "exit_status": "Submitted",
            "submission": "Fixed the issue",
            "model_stats": {"instance_cost": 0.05, "api_calls": 3},
        },
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": [{"type": "text", "text": "Please solve this issue."}]},
            {"role": "assistant", "content": "I'll analyze the issue.\n\n```bash\ncat file.py\n```"},
            {"role": "user", "content": [{"type": "text", "text": "File contents here."}]},
            {"role": "assistant", "content": "Fixed!\n\n```bash\necho MICRO_SWE_AGENT_FINAL_OUTPUT\n```"},
        ],
    }


@pytest.fixture
def temp_trajectory_files(sample_simple_trajectory, sample_swebench_trajectory):
    """Create temporary trajectory files for testing."""
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)

        # Simple format trajectory
        simple_file = temp_path / "simple.traj.json"
        simple_file.write_text(json.dumps(sample_simple_trajectory, indent=2))

        # SWEBench format trajectory
        swebench_file = temp_path / "swebench.traj.json"
        swebench_file.write_text(json.dumps(sample_swebench_trajectory, indent=2))

        # Invalid JSON file
        invalid_file = temp_path / "invalid.traj.json"
        invalid_file.write_text("invalid json content")

        yield [simple_file, swebench_file, invalid_file]


@pytest.mark.slow
async def test_trajectory_inspector_basic_navigation(temp_trajectory_files):
    """Test basic step navigation in trajectory inspector."""
    valid_files = [f for f in temp_trajectory_files if f.name != "invalid.traj.json"]

    app = TrajectoryInspector(valid_files)

    async with app.run_test() as pilot:
        # Should start with first trajectory, first step
        await pilot.pause(0.1)
        assert "[1/2] simple.traj.json - Step 1/3" in app.title
        content = get_screen_text(app)
        assert "SYSTEM" in content
        assert "You are a helpful assistant" in content
        assert "solve this problem" in content

        # Navigate to next step
        await pilot.press("l")
        assert "Step 2/3" in app.title
        assert "MICRO-SWE-AGENT" in get_screen_text(app)
        assert "I'll help you solve this" in get_screen_text(app)

        # Navigate to last step
        await pilot.press("$")
        assert "Step 3/3" in app.title
        assert "MICRO-SWE-AGENT" in get_screen_text(app)
        assert "echo MICRO_SWE_AGENT_FINAL_OUTPUT" in get_screen_text(app)

        # Navigate back to first step
        await pilot.press("0")
        assert "Step 1/3" in app.title
        assert "SYSTEM" in get_screen_text(app)

        # Navigate with left/right arrows
        await pilot.press("right")
        assert "Step 2/3" in app.title
        await pilot.press("left")
        assert "Step 1/3" in app.title


@pytest.mark.slow
async def test_trajectory_inspector_trajectory_navigation(temp_trajectory_files):
    """Test navigation between different trajectory files."""
    valid_files = [f for f in temp_trajectory_files if f.name != "invalid.traj.json"]

    app = TrajectoryInspector(valid_files)

    async with app.run_test() as pilot:
        await pilot.pause(0.1)

        # Should start with first trajectory
        assert "[1/2] simple.traj.json" in app.title
        content = get_screen_text(app)
        assert "You are a helpful assistant" in content

        # Navigate to next trajectory
        await pilot.press("L")
        assert "[2/2] swebench.traj.json" in app.title
        await pilot.pause(0.1)
        content = get_screen_text(app)
        assert "You are a helpful assistant" in content

        # Navigate back to previous trajectory
        await pilot.press("H")
        assert "[1/2] simple.traj.json" in app.title

        # Try to navigate beyond bounds
        await pilot.press("H")  # Should stay at first
        assert "[1/2] simple.traj.json" in app.title

        await pilot.press("L")  # Go to second
        await pilot.press("L")  # Try to go beyond
        assert "[2/2] swebench.traj.json" in app.title  # Should stay at last


@pytest.mark.slow
async def test_trajectory_inspector_swebench_format(temp_trajectory_files):
    """Test that SWEBench format trajectories are handled correctly."""
    valid_files = [f for f in temp_trajectory_files if f.name != "invalid.traj.json"]

    app = TrajectoryInspector(valid_files)

    async with app.run_test() as pilot:
        # Navigate to SWEBench trajectory
        await pilot.press("L")
        await pilot.pause(0.1)

        assert "[2/2] swebench.traj.json" in app.title
        assert "Step 1/3" in app.title

        # Check that list content is properly rendered - step 1 should have the initial user message
        content = get_screen_text(app)
        assert "Please solve this issue" in content


@pytest.mark.slow
async def test_trajectory_inspector_scrolling(temp_trajectory_files):
    """Test scrolling behavior in trajectory inspector."""
    valid_files = [f for f in temp_trajectory_files if f.name != "invalid.traj.json"]

    app = TrajectoryInspector(valid_files)

    async with app.run_test() as pilot:
        await pilot.pause(0.1)

        # Test scrolling
        vs = app.query_one("VerticalScroll")
        initial_y = vs.scroll_target_y

        await pilot.press("j")  # scroll down
        assert vs.scroll_target_y >= initial_y

        await pilot.press("k")  # scroll up
        # Should scroll up (may not be exactly equal due to content constraints)


@pytest.mark.slow
async def test_trajectory_inspector_empty_trajectory():
    """Test inspector behavior with empty trajectory list."""
    app = TrajectoryInspector([])

    async with app.run_test() as pilot:
        await pilot.pause(0.1)

        assert "Trajectory Inspector - No Data" in app.title
        assert "No trajectory loaded" in get_screen_text(app)

        # Navigation should not crash
        await pilot.press("l")
        await pilot.press("h")
        await pilot.press("L")
        await pilot.press("H")


async def test_trajectory_inspector_invalid_file(temp_trajectory_files):
    """Test inspector behavior with invalid JSON file."""
    invalid_file = [f for f in temp_trajectory_files if f.name == "invalid.traj.json"][0]

    # Mock notify to capture error messages
    app = TrajectoryInspector([invalid_file])

    # Since this is not an async run_test, we need to manually trigger the load
    # The error should be captured when _load_current_trajectory is called
    app._load_current_trajectory()

    assert app.messages == []
    assert app.steps == []


def test_trajectory_inspector_load_trajectory_formats(sample_simple_trajectory, sample_swebench_trajectory):
    """Test loading different trajectory formats."""
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)

        # Test simple format
        simple_file = temp_path / "simple.traj.json"
        simple_file.write_text(json.dumps(sample_simple_trajectory))

        app = TrajectoryInspector([simple_file])
        assert len(app.messages) == 5
        assert len(app.steps) == 3

        # Test SWEBench format
        swebench_file = temp_path / "swebench.traj.json"
        swebench_file.write_text(json.dumps(sample_swebench_trajectory))

        app = TrajectoryInspector([swebench_file])
        assert len(app.messages) == 5
        assert len(app.steps) == 3


def test_trajectory_inspector_unrecognized_format():
    """Test inspector behavior with unrecognized trajectory format."""
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)

        # Create file with unrecognized format
        unrecognized_file = temp_path / "unrecognized.traj.json"
        unrecognized_file.write_text(json.dumps({"some": "other", "format": True}))

        app = TrajectoryInspector([unrecognized_file])

        # Should handle gracefully
        assert app.messages == []
        assert app.steps == []


def test_trajectory_inspector_current_trajectory_name():
    """Test current_trajectory_name property."""
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        test_file = temp_path / "test.traj.json"
        test_file.write_text(json.dumps([]))

        app = TrajectoryInspector([test_file])
        assert app.current_trajectory_name == "test.traj.json"

        # Test with empty trajectory list
        app = TrajectoryInspector([])
        assert app.current_trajectory_name == "No trajectories"


@pytest.mark.slow
async def test_trajectory_inspector_css_loading():
    """Test that CSS is properly loaded from config."""
    app = TrajectoryInspector([])

    # Verify CSS contains expected styles
    assert ".message-container" in app.CSS
    assert ".message-header" in app.CSS
    assert ".message-content" in app.CSS


@pytest.mark.slow
async def test_trajectory_inspector_quit_binding(temp_trajectory_files):
    """Test quit functionality."""
    valid_files = [f for f in temp_trajectory_files if f.name != "invalid.traj.json"]

    app = TrajectoryInspector(valid_files)

    async with app.run_test() as pilot:
        await pilot.pause(0.1)

        # Test quit functionality
        await pilot.press("q")
        await pilot.pause(0.1)

        # App should exit gracefully (the test framework handles this)

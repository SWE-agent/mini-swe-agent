"""Tests for HarborMiniAgent - agent with incremental trajectory saving."""

import json
import tempfile
from pathlib import Path

from minisweagent.agents.harbor import HarborMiniAgent
from minisweagent.environments.local import LocalEnvironment
from minisweagent.models.test_models import DeterministicModel


def test_saves_trajectory_after_each_step():
    """Test that trajectory is saved after each step."""
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        traj_path = Path(f.name)

    try:
        agent = HarborMiniAgent(
            model=DeterministicModel(
                outputs=[
                    "Step 1\n```bash\necho 'first'\n```",
                    "Step 2\n```bash\necho 'COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT'\necho 'done'\n```",
                ]
            ),
            env=LocalEnvironment(),
            traj_path=traj_path,
        )

        exit_status, result = agent.run("Test incremental save")
        assert exit_status == "Submitted"
        assert result == "done\n"

        # Verify trajectory was saved
        assert traj_path.exists()
        data = json.loads(traj_path.read_text())
        assert "messages" in data
        assert len(data["messages"]) == 5  # system, user, assistant, user, assistant
    finally:
        traj_path.unlink(missing_ok=True)


def test_trajectory_saved_even_on_error():
    """Test that trajectory is saved even when an error occurs."""
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        traj_path = Path(f.name)

    try:
        agent = HarborMiniAgent(
            model=DeterministicModel(
                outputs=[
                    "No code block here",  # This will cause a format error
                    "Fix it\n```bash\necho 'COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT'\necho 'recovered'\n```",
                ]
            ),
            env=LocalEnvironment(),
            traj_path=traj_path,
        )

        exit_status, result = agent.run("Test error handling")
        assert exit_status == "Submitted"
        assert result == "recovered\n"

        # Verify trajectory was saved
        assert traj_path.exists()
        data = json.loads(traj_path.read_text())
        assert len(data["messages"]) > 0
    finally:
        traj_path.unlink(missing_ok=True)


def test_no_trajectory_saved_without_path():
    """Test that no trajectory is saved when traj_path is None."""
    agent = HarborMiniAgent(
        model=DeterministicModel(outputs=["```bash\necho 'COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT'\necho 'done'\n```"]),
        env=LocalEnvironment(),
        traj_path=None,
    )

    exit_status, result = agent.run("Test no save")
    assert exit_status == "Submitted"
    assert result == "done\n"
    # No error should occur, just no file saved


def test_inherits_default_agent_behavior():
    """Test that HarborMiniAgent inherits all DefaultAgent behavior."""
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        traj_path = Path(f.name)

    try:
        agent = HarborMiniAgent(
            model=DeterministicModel(
                outputs=["Response\n```bash\necho 'COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT'\necho 'works'\n```"]
            ),
            env=LocalEnvironment(),
            traj_path=traj_path,
            system_template="Custom system prompt",
            cost_limit=5.0,
        )

        exit_status, result = agent.run("Test inheritance")
        assert exit_status == "Submitted"
        assert result == "works\n"
        assert agent.messages[0]["content"] == "Custom system prompt"
    finally:
        traj_path.unlink(missing_ok=True)

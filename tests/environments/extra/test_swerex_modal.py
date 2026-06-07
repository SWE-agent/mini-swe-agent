from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from minisweagent.environments.extra.swerex_modal import (
    SwerexModalEnvironment,
    SwerexModalEnvironmentConfig,
)
from minisweagent.exceptions import Submitted


def _make_env(**kwargs):
    """Create a SwerexModalEnvironment with mocked __init__ (no Modal infra)."""
    with patch.object(SwerexModalEnvironment, "__init__", lambda self, **kw: None):
        env = SwerexModalEnvironment()
        env.config = SwerexModalEnvironmentConfig(image="python:3.11", **kwargs)
        return env


def test_swerex_modal_serialize():
    """Test that SwerexModalEnvironment.serialize() returns the expected structure."""
    env = _make_env()
    result = env.serialize()

    assert "info" in result
    assert "config" in result["info"]
    assert "environment" in result["info"]["config"]
    assert "environment_type" in result["info"]["config"]
    assert result["info"]["config"]["environment"]["image"] == "python:3.11"
    assert "SwerexModalEnvironment" in result["info"]["config"]["environment_type"]


def test_swerex_modal_execute_accepts_dict_action():
    """Test that execute() accepts v2 dict action format."""
    env = _make_env()

    mock_output = MagicMock()
    mock_output.stdout = "hello world\n"
    mock_output.exit_code = 0

    mock_runtime = MagicMock()
    mock_runtime.execute = AsyncMock(return_value=mock_output)

    mock_deployment = MagicMock()
    mock_deployment.runtime = mock_runtime
    env.deployment = mock_deployment

    result = env.execute({"command": "echo hello world"})

    assert result["output"] == "hello world\n"
    assert result["returncode"] == 0


def test_swerex_modal_execute_raises_submitted():
    """Test that execute() raises Submitted when output contains the submission marker."""
    env = _make_env()

    mock_output = MagicMock()
    mock_output.stdout = "COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT\ndiff --git a/file.py b/file.py\n"
    mock_output.exit_code = 0

    mock_runtime = MagicMock()
    mock_runtime.execute = AsyncMock(return_value=mock_output)

    mock_deployment = MagicMock()
    mock_deployment.runtime = mock_runtime
    env.deployment = mock_deployment

    with pytest.raises(Submitted) as exc_info:
        env.execute({"command": "echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT && git diff"})

    msg = exc_info.value.messages[0]
    assert msg["extra"]["exit_status"] == "Submitted"
    assert "diff --git" in msg["extra"]["submission"]

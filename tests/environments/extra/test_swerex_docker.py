import pytest

from minisweagent.environments.extra.swerex_docker import SwerexDockerEnvironment


@pytest.mark.slow
def test_swerex_docker_basic_execution():
    """Test basic command execution in SwerexDockerEnvironment."""
    env = SwerexDockerEnvironment(image="python:3.11")

    result = env.execute("echo 'hello world'")

    assert isinstance(result, dict)
    assert "output" in result
    assert "returncode" in result
    assert result["returncode"] == 0
    assert "hello world" in result["output"]


@pytest.mark.slow
def test_swerex_docker_command_failure():
    """Test that command failures are properly captured in SwerexDockerEnvironment."""
    env = SwerexDockerEnvironment(image="python:3.11")

    result = env.execute("exit 1")

    assert isinstance(result, dict)
    assert "output" in result
    assert "returncode" in result
    assert result["returncode"] == 1

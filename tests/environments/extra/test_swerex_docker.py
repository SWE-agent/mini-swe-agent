import subprocess

import pytest

from minisweagent.environments.extra.swerex_docker import SwerexDockerEnvironment


def _is_docker_available():
    """Check if Docker (or podman aliased as docker) is available."""
    try:
        subprocess.run(["docker", "version"], capture_output=True, check=True, timeout=5)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired):
        return False


@pytest.mark.slow
@pytest.mark.skipif(not _is_docker_available(), reason="Docker not available (swerex requires docker)")
def test_swerex_docker_basic_execution():
    """Test basic command execution in SwerexDockerEnvironment."""
    env = SwerexDockerEnvironment(image="python:3.11")

    result = env.execute({"command": "echo 'hello world'"})

    assert isinstance(result, dict)
    assert "output" in result
    assert "returncode" in result
    assert result["returncode"] == 0
    assert "hello world" in result["output"]


@pytest.mark.slow
@pytest.mark.skipif(not _is_docker_available(), reason="Docker not available (swerex requires docker)")
def test_swerex_docker_command_failure():
    """Test that command failures are properly captured in SwerexDockerEnvironment."""
    env = SwerexDockerEnvironment(image="python:3.11")

    result = env.execute({"command": "exit 1"})

    assert isinstance(result, dict)
    assert "output" in result
    assert "returncode" in result
    assert result["returncode"] == 1

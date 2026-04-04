import os
import subprocess
from unittest.mock import MagicMock, patch

import pytest

from minisweagent.environments.docker import DockerEnvironment, DockerEnvironmentConfig
from minisweagent.exceptions import ContainerNotRunning


def is_docker_available():
    """Check if Docker is available and running."""
    try:
        subprocess.run(["docker", "version"], capture_output=True, check=True, timeout=5)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired):
        return False


def is_podman_available():
    """Check if Podman is available and running."""
    try:
        subprocess.run(["podman", "version"], capture_output=True, check=True, timeout=5)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired):
        return False


# Test parameters for both Docker and Podman
environment_params = [
    pytest.param(
        "docker",
        marks=pytest.mark.skipif(not is_docker_available(), reason="Docker not available"),
        id="docker",
    ),
    pytest.param(
        "podman",
        marks=pytest.mark.skipif(not is_podman_available(), reason="Podman not available"),
        id="podman",
    ),
]


@pytest.mark.parametrize("executable", environment_params)
def test_docker_environment_config_defaults(executable):
    """Test that DockerEnvironmentConfig has correct default values."""
    config = DockerEnvironmentConfig(image="python:3.11", executable=executable)

    assert config.image == "python:3.11"
    assert config.cwd == "/"
    assert config.env == {}
    assert config.forward_env == []
    assert config.timeout == 30
    assert config.executable == executable


@pytest.mark.slow
@pytest.mark.parametrize("executable", environment_params)
def test_docker_environment_basic_execution(executable):
    """Test basic command execution in Docker container."""
    env = DockerEnvironment(image="python:3.11", executable=executable)

    try:
        result = env.execute({"command": "echo 'hello world'"})
        assert result["returncode"] == 0
        assert "hello world" in result["output"]
    finally:
        env.cleanup()


@pytest.mark.slow
@pytest.mark.parametrize("executable", environment_params)
def test_docker_environment_set_env_variables(executable):
    """Test setting environment variables in the container."""
    env = DockerEnvironment(
        image="python:3.11", executable=executable, env={"TEST_VAR": "test_value", "ANOTHER_VAR": "another_value"}
    )

    try:
        # Test single environment variable
        result = env.execute({"command": "echo $TEST_VAR"})
        assert result["returncode"] == 0
        assert "test_value" in result["output"]

        # Test multiple environment variables
        result = env.execute({"command": "echo $TEST_VAR $ANOTHER_VAR"})
        assert result["returncode"] == 0
        assert "test_value another_value" in result["output"]
    finally:
        env.cleanup()


@pytest.mark.slow
@pytest.mark.parametrize("executable", environment_params)
def test_docker_environment_forward_env_variables(executable):
    """Test forwarding environment variables from host to container."""
    with patch.dict(os.environ, {"HOST_VAR": "host_value", "ANOTHER_HOST_VAR": "another_host_value"}):
        env = DockerEnvironment(
            image="python:3.11", executable=executable, forward_env=["HOST_VAR", "ANOTHER_HOST_VAR"]
        )

        try:
            # Test single forwarded environment variable
            result = env.execute({"command": "echo $HOST_VAR"})
            assert result["returncode"] == 0
            assert "host_value" in result["output"]

            # Test multiple forwarded environment variables
            result = env.execute({"command": "echo $HOST_VAR $ANOTHER_HOST_VAR"})
            assert result["returncode"] == 0
            assert "host_value another_host_value" in result["output"]
        finally:
            env.cleanup()


@pytest.mark.slow
@pytest.mark.parametrize("executable", environment_params)
def test_docker_environment_forward_nonexistent_env_variables(executable):
    """Test forwarding non-existent environment variables (should be empty)."""
    env = DockerEnvironment(image="python:3.11", executable=executable, forward_env=["NONEXISTENT_VAR"])

    try:
        result = env.execute({"command": 'echo "[$NONEXISTENT_VAR]"'})
        assert result["returncode"] == 0
        assert "[]" in result["output"]  # Empty variable should result in empty string
    finally:
        env.cleanup()


@pytest.mark.slow
@pytest.mark.parametrize("executable", environment_params)
def test_docker_environment_combined_env_and_forward(executable):
    """Test both setting and forwarding environment variables together."""
    with patch.dict(os.environ, {"HOST_VAR": "from_host"}):
        env = DockerEnvironment(
            image="python:3.11", executable=executable, env={"SET_VAR": "from_config"}, forward_env=["HOST_VAR"]
        )

        try:
            result = env.execute({"command": "echo $SET_VAR $HOST_VAR"})
            assert result["returncode"] == 0
            assert "from_config from_host" in result["output"]
        finally:
            env.cleanup()


@pytest.mark.slow
@pytest.mark.parametrize("executable", environment_params)
def test_docker_environment_env_override_forward(executable):
    """Test that explicitly set env variables take precedence over forwarded ones."""
    with patch.dict(os.environ, {"CONFLICT_VAR": "from_host"}):
        env = DockerEnvironment(
            image="python:3.11",
            executable=executable,
            env={"CONFLICT_VAR": "from_config"},
            forward_env=["CONFLICT_VAR"],
        )

        try:
            result = env.execute({"command": "echo $CONFLICT_VAR"})
            assert result["returncode"] == 0
            # The explicitly set env should take precedence (comes first in docker exec command)
            assert "from_config" in result["output"]
        finally:
            env.cleanup()


@pytest.mark.slow
@pytest.mark.parametrize("executable", environment_params)
def test_docker_environment_custom_cwd(executable):
    """Test executing commands in a custom working directory."""
    env = DockerEnvironment(image="python:3.11", executable=executable, cwd="/tmp")

    try:
        result = env.execute({"command": "pwd"})
        assert result["returncode"] == 0
        assert "/tmp" in result["output"]
    finally:
        env.cleanup()


@pytest.mark.slow
@pytest.mark.parametrize("executable", environment_params)
def test_docker_environment_cwd_parameter_override(executable):
    """Test that the cwd parameter in execute() overrides the config cwd."""
    env = DockerEnvironment(image="python:3.11", executable=executable, cwd="/")

    try:
        result = env.execute({"command": "pwd"}, cwd="/tmp")
        assert result["returncode"] == 0
        assert "/tmp" in result["output"]
    finally:
        env.cleanup()


@pytest.mark.slow
@pytest.mark.parametrize("executable", environment_params)
def test_docker_environment_command_failure(executable):
    """Test that command failures are properly captured."""
    env = DockerEnvironment(image="python:3.11", executable=executable)

    try:
        result = env.execute({"command": "exit 42"})
        assert result["returncode"] == 42
    finally:
        env.cleanup()


@pytest.mark.slow
@pytest.mark.parametrize("executable", environment_params)
def test_docker_environment_custom_container_timeout(executable):
    """Test that custom container_timeout is respected."""
    import time

    env = DockerEnvironment(image="python:3.11", executable=executable, container_timeout="3s")

    try:
        result = env.execute({"command": "echo 'container is running'"})
        assert result["returncode"] == 0
        assert "container is running" in result["output"]
        time.sleep(5)
        with pytest.raises((subprocess.CalledProcessError, subprocess.TimeoutExpired)):
            # This command should fail because the container has stopped
            subprocess.run(
                [executable, "exec", env.container_id, "echo", "still running"],
                check=True,
                capture_output=True,
                timeout=2,
            )
    finally:
        env.cleanup()


@pytest.mark.slow
@pytest.mark.parametrize("executable", environment_params)
def test_docker_environment_raises_container_not_running_after_timeout(executable):
    """Test that ContainerNotRunning is raised when executing after container_timeout expires.

    Regression test for https://github.com/SWE-agent/mini-swe-agent/issues/803.
    Previously, the agent would silently continue making API calls after the container died.
    """
    import time

    env = DockerEnvironment(image="python:3.11", executable=executable, container_timeout="3s")

    try:
        # Container should work initially
        result = env.execute({"command": "echo 'alive'"})
        assert result["returncode"] == 0

        # Wait for container to die
        time.sleep(5)

        # Executing after container death should raise ContainerNotRunning
        with pytest.raises(ContainerNotRunning) as exc_info:
            env.execute({"command": "echo 'should fail'"})

        # Verify the exception contains proper exit info
        assert len(exc_info.value.messages) == 1
        msg = exc_info.value.messages[0]
        assert msg["role"] == "exit"
        assert msg["extra"]["exit_status"] == "ContainerNotRunning"
        assert msg["extra"]["submission"] == ""
    finally:
        env.cleanup()


def test_docker_environment_detects_dead_container_from_returncode():
    """Unit test: ContainerNotRunning raised when subprocess output contains dead-container markers."""
    mock_result = MagicMock()
    mock_result.stdout = "Error response from daemon: No such container: abc123\n"
    mock_result.returncode = 1

    with patch("minisweagent.environments.docker.subprocess") as mock_subprocess:
        # Mock the container start
        mock_start = MagicMock()
        mock_start.stdout = "fake_container_id\n"
        mock_subprocess.run.side_effect = [mock_start, mock_result]
        mock_subprocess.PIPE = subprocess.PIPE
        mock_subprocess.STDOUT = subprocess.STDOUT

        env = DockerEnvironment(image="python:3.11")

        with pytest.raises(ContainerNotRunning) as exc_info:
            env.execute({"command": "echo test"})

        msg = exc_info.value.messages[0]
        assert msg["role"] == "exit"
        assert msg["extra"]["exit_status"] == "ContainerNotRunning"


def test_docker_environment_detects_dead_container_from_exception():
    """Unit test: ContainerNotRunning raised when subprocess raises with dead-container message."""
    with patch("minisweagent.environments.docker.subprocess") as mock_subprocess:
        # Mock the container start
        mock_start = MagicMock()
        mock_start.stdout = "fake_container_id\n"
        mock_subprocess.PIPE = subprocess.PIPE
        mock_subprocess.STDOUT = subprocess.STDOUT

        # First call starts container, second call raises with "is not running"
        mock_subprocess.run.side_effect = [
            mock_start,
            Exception("Error: container fake_container_id is not running"),
        ]

        env = DockerEnvironment(image="python:3.11")

        with pytest.raises(ContainerNotRunning) as exc_info:
            env.execute({"command": "echo test"})

        msg = exc_info.value.messages[0]
        assert msg["role"] == "exit"
        assert msg["extra"]["exit_status"] == "ContainerNotRunning"

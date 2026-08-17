import os
import subprocess
from unittest.mock import patch

import pytest

from minisweagent.environments.docker import DockerEnvironment, DockerEnvironmentConfig
from minisweagent.exceptions import Submitted


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


def _check_finished(output: dict):
    # _check_finished only reads the output dict, so no container is needed.
    return DockerEnvironment.__new__(DockerEnvironment)._check_finished(output)


@pytest.mark.parametrize(
    ("stdout", "expected_submission"),
    [
        ("COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT\ndiff --git a/f b/f\n", "diff --git a/f b/f\n"),
        # Startup noise (e.g. a BASH_ENV sourcing a missing conda path) pushes the marker off line 0 (#938).
        (
            "/root/.bashrc: line 1: /opt/miniconda3/etc/profile.d/conda.sh: No such file or directory\n"
            "COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT\ndiff --git a/f b/f\n",
            "diff --git a/f b/f\n",
        ),
    ],
)
def test_check_finished_detects_marker_past_startup_noise(stdout, expected_submission):
    with pytest.raises(Submitted) as exc_info:
        _check_finished({"output": stdout, "returncode": 0})
    assert exc_info.value.messages[0]["extra"]["submission"] == expected_submission


@pytest.mark.parametrize(
    ("stdout", "returncode"),
    [
        ("COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT\npatch\n", 1),  # marker present but command failed
        ("regular command output\nnothing to submit\n", 0),  # no marker
    ],
)
def test_check_finished_does_not_submit(stdout, returncode):
    assert _check_finished({"output": stdout, "returncode": returncode}) is None

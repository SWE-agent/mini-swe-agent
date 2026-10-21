import logging
import os
import subprocess
import uuid
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
    assert config.terminate_process_group_on_timeout is True
    assert config.termination_grace_period == 1.0
    assert config.cleanup_timeout == 15


def make_unstarted_environment(**kwargs):
    """Build a DockerEnvironment without starting a container for focused unit tests."""
    env = object.__new__(DockerEnvironment)
    env.logger = logging.getLogger("test.docker")
    env.container_id = "container-id"
    env.config = DockerEnvironmentConfig(image="python:3.11", **kwargs)
    return env


def test_submission_marker_after_leading_output_is_detected():
    env = make_unstarted_environment()

    with pytest.raises(Submitted) as exc_info:
        env._check_finished(
            {
                "output": "running checks\nCOMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT\npatch contents\n",
                "returncode": 0,
            }
        )

    assert exc_info.value.messages[0]["extra"]["submission"] == "patch contents\n"


def test_cleanup_is_synchronous_cross_platform_and_idempotent():
    env = make_unstarted_environment(cleanup_timeout=3)
    completed = subprocess.CompletedProcess(args=[], returncode=0)

    with patch("minisweagent.environments.docker.subprocess.run", return_value=completed) as run:
        env.cleanup()
        env.cleanup()

    run.assert_called_once_with(
        ["docker", "stop", "--time", "10", "container-id"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        timeout=3,
    )
    assert env.container_id is None


def test_cleanup_force_removes_container_when_stop_fails():
    env = make_unstarted_environment(cleanup_timeout=3)
    stop_failed = subprocess.CompletedProcess(args=[], returncode=1)
    removed = subprocess.CompletedProcess(args=[], returncode=0)

    with patch("minisweagent.environments.docker.subprocess.run", side_effect=[stop_failed, removed]) as run:
        env.cleanup()

    assert run.call_args_list[1].args[0] == ["docker", "rm", "-f", "container-id"]
    assert env.container_id is None


def test_timeout_triggers_container_side_process_group_cleanup():
    env = make_unstarted_environment(timeout=1, termination_grace_period=0)
    timed_out = subprocess.TimeoutExpired(cmd=["docker", "exec"], timeout=1, output="partial output")
    cleaned = subprocess.CompletedProcess(args=[], returncode=0, stdout="", stderr="")

    with patch("minisweagent.environments.docker.subprocess.run", side_effect=[timed_out, cleaned]) as run:
        result = env.execute({"command": "sleep 100"})

    assert result["returncode"] == -1
    assert result["extra"]["exception_type"] == "TimeoutExpired"
    execute_command = run.call_args_list[0].args[0]
    assert "setsid bash -lc" in execute_command[-1]
    cleanup_command = run.call_args_list[1].args[0]
    assert cleanup_command[:4] == ["docker", "exec", "container-id", "bash"]
    assert "kill -TERM" in cleanup_command[-1]


@pytest.mark.slow
@pytest.mark.parametrize("executable", environment_params)
def test_timeout_leaves_no_container_side_processes(executable):
    marker = f"minisweagent_timeout_{uuid.uuid4().hex}"
    env = DockerEnvironment(
        image="python:3.11",
        executable=executable,
        timeout=1,
        termination_grace_period=0.1,
    )

    try:
        result = env.execute(
            {
                "command": (
                    "python3 -c 'import signal,time; "
                    "signal.signal(signal.SIGTERM,signal.SIG_IGN); "
                    f"time.sleep(60)' {marker} & wait"
                )
            }
        )
        processes = subprocess.run(
            [
                executable,
                "exec",
                env.container_id,
                "bash",
                "-lc",
                f"pgrep -af {marker!r} | grep -v pgrep || true",
            ],
            capture_output=True,
            text=True,
            timeout=5,
            check=True,
        )

        assert result["extra"]["exception_type"] == "TimeoutExpired"
        assert processes.stdout.strip() == ""
    finally:
        env.cleanup()


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

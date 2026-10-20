import json
import logging
import os
import subprocess
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from minisweagent.environments.docker import MARKER, DockerEnvironment, DockerEnvironmentConfig
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


def _make_stub_env(**config_overrides) -> DockerEnvironment:
    env = DockerEnvironment.__new__(DockerEnvironment)
    env.logger = logging.getLogger("test_docker")
    env.container_id = "test-container"
    defaults = {
        "image": "python:3.11",
        "timeout": 1,
        "terminate_grace_seconds": 0.2,
        "executable": "docker",
        "cwd": "/tmp",
        "forward_env": [],
        "env": {},
        "interpreter": ["bash", "-c"],
    }
    defaults.update(config_overrides)
    env.config = DockerEnvironmentConfig(**defaults)
    return env


_SUBMISSION_MARKER = "COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT"
_DIFF = "diff --git a/file b/file\n--- a/file\n+++ b/file\n@@ -1 +1 @@\n-old\n+new\n"
_PREVIEW = "diff --git a/file b/file\n--- a/file\n+++ b/file\n"


@pytest.mark.parametrize(
    ("stdout", "rc", "timed_out", "want"),
    [
        (_PREVIEW + _SUBMISSION_MARKER + "\n" + _DIFF, 0, False, _DIFF),
        (_SUBMISSION_MARKER + "\n" + _DIFF, 0, False, _DIFF),
        (
            "progress\r\n" + _SUBMISSION_MARKER + "\r\n" + _DIFF.replace("\n", "\r\n"),
            0,
            False,
            _DIFF.replace("\n", "\r\n"),
        ),
        ("progress\n" + _SUBMISSION_MARKER, 0, False, ""),
        ("progress\n" + _SUBMISSION_MARKER + "\n" + _DIFF, 1, False, None),
        ("progress\n" + _SUBMISSION_MARKER + "\n" + _DIFF, -15, True, None),
        ("echo " + _SUBMISSION_MARKER + "\n" + _DIFF, 0, False, None),
        (_SUBMISSION_MARKER + " extra\n" + _DIFF, 0, False, None),
        ("progress\n " + _SUBMISSION_MARKER + "\n" + _DIFF, 0, False, None),
        ("progress\n" + _SUBMISSION_MARKER + " \n" + _DIFF, 0, False, None),
        (" " + _SUBMISSION_MARKER + "\n" + _DIFF, 0, False, None),
        (_SUBMISSION_MARKER + " \n" + _DIFF, 0, False, None),
        ("ordinary output\n", 0, False, None),
        ("", 0, False, None),
    ],
)
def test_docker_environment_submission_detection(stdout, rc, timed_out, want):
    """Submits only output after an exact marker line when exit code is 0."""
    env = _make_stub_env()
    encoded = json.dumps({"returncode": rc, "timed_out": timed_out})
    proc_result = SimpleNamespace(returncode=0, stdout=f"{stdout}\n{MARKER}{encoded}")
    try:
        with patch("minisweagent.environments.docker.subprocess.run", return_value=proc_result):
            if want is not None:
                with pytest.raises(Submitted) as exc_info:
                    env.execute({"command": "test"})
                assert exc_info.value.messages[0] == {
                    "role": "exit",
                    "content": want,
                    "extra": {"exit_status": "Submitted", "submission": want},
                }
            else:
                observation = env.execute({"command": "test"})
                assert observation["output"] == stdout
                assert observation["returncode"] == (-1 if timed_out else rc)
                assert bool(observation["exception_info"]) == timed_out
    finally:
        env.container_id = None


@pytest.mark.parametrize(
    ("timeout", "grace"),
    [
        (0, 1.0),
        (-1, 1.0),
        (float("inf"), 1.0),
        (1, 0.0),
        (1, 6.0),
        (1, float("nan")),
    ],
)
def test_docker_environment_execute_validates_bounds(timeout, grace):
    env = _make_stub_env(terminate_grace_seconds=grace)
    try:
        with pytest.raises(ValueError, match="Command timeout must be positive"):
            env.execute({"command": "true"}, timeout=timeout)
    finally:
        env.container_id = None


def test_docker_environment_execute_requires_running_container():
    env = _make_stub_env()
    env.container_id = None
    assert env._is_container_running() is False
    with pytest.raises(RuntimeError, match="Task container is not running"):
        env.execute({"command": "true"})


def test_docker_environment_is_container_running_branches():
    env = _make_stub_env()
    try:
        with patch(
            "minisweagent.environments.docker.subprocess.run",
            return_value=SimpleNamespace(returncode=0, stdout="true\n"),
        ):
            assert env._is_container_running() is True
        with patch(
            "minisweagent.environments.docker.subprocess.run",
            return_value=SimpleNamespace(returncode=1, stdout="false\n"),
        ):
            assert env._is_container_running() is False
        with patch(
            "minisweagent.environments.docker.subprocess.run",
            side_effect=subprocess.TimeoutExpired("docker", 5),
        ):
            assert env._is_container_running() is False
    finally:
        env.container_id = None


def test_docker_environment_oom_killed_detection():
    env = _make_stub_env()
    try:
        calls = [
            SimpleNamespace(returncode=137, stdout="killed output"),
            SimpleNamespace(returncode=0, stdout="true\n"),
        ]
        with patch("minisweagent.environments.docker.subprocess.run", side_effect=calls):
            obs = env.execute({"command": "alloc"})
        assert obs["returncode"] == -9
        assert obs["output"] == "killed output"
        assert "SIGKILL" in obs["exception_info"]

        encoded = json.dumps({"returncode": 137, "timed_out": False})
        with patch(
            "minisweagent.environments.docker.subprocess.run",
            return_value=SimpleNamespace(returncode=0, stdout=f"child oom\n{MARKER}{encoded}"),
        ):
            obs2 = env.execute({"command": "alloc2"})
        assert obs2["returncode"] == -9
        assert "SIGKILL" in obs2["exception_info"]
    finally:
        env.container_id = None


def test_docker_environment_supervisor_failure_cleans_up():
    env = _make_stub_env()
    calls = [
        SimpleNamespace(returncode=1, stdout="missing python3"),
        SimpleNamespace(returncode=0, stderr=""),
    ]
    with patch("minisweagent.environments.docker.subprocess.run", side_effect=calls):
        with pytest.raises(RuntimeError, match="Command supervisor did not confirm termination"):
            env.execute({"command": "true"})
    assert env.container_id is None


def test_docker_environment_invalid_supervisor_payload_and_cleanup_error():
    env = _make_stub_env()
    bad_payload = json.dumps({"returncode": "bad", "timed_out": "no"})
    calls = [
        SimpleNamespace(returncode=0, stdout=f"out\n{MARKER}{bad_payload}"),
        SimpleNamespace(returncode=1, stderr="daemon unreachable"),
    ]
    try:
        with patch("minisweagent.environments.docker.subprocess.run", side_effect=calls):
            with pytest.raises(RuntimeError, match="Invalid command supervisor result; Task container cleanup failed"):
                env.execute({"command": "true"})
    finally:
        env.container_id = None


def test_docker_environment_cleanup_and_del_edge_cases():
    env = _make_stub_env()
    env.container_id = None
    env.cleanup()

    env.container_id = "gone-id"
    with patch(
        "minisweagent.environments.docker.subprocess.run",
        return_value=SimpleNamespace(returncode=1, stderr="Error: No such container: gone-id"),
    ):
        env.cleanup()
    assert env.container_id is None

    env.container_id = "stuck-id"
    with patch(
        "minisweagent.environments.docker.subprocess.run",
        return_value=SimpleNamespace(returncode=1, stderr="permission denied"),
    ):
        env.__del__()
    env.container_id = None


@pytest.mark.slow
@pytest.mark.parametrize("executable", environment_params)
def test_docker_environment_containment_and_process_group_kill(executable):
    """Kills loops and TERM-resistant children on timeout and bounds container memory."""
    import time

    env = DockerEnvironment(
        image="python:3.11",
        executable=executable,
        cwd="/tmp",
        timeout=1,
        terminate_grace_seconds=0.2,
        run_args=["--rm", "--init"],
    )
    try:
        assert env.execute({"command": "printf 'hello'; exit 7"})["returncode"] == 7
        assert env.execute({"command": "printf 'hello'"})["output"] == "hello"
        for command in [
            "python3 -c 'while True: pass'",
            (
                'python3 -c "import os, pathlib, signal, time; '
                "signal.signal(signal.SIGTERM, signal.SIG_IGN); "
                "pathlib.Path('/tmp/loop.pid').write_text(str(os.getpid())); "
                'time.sleep(100)" & wait'
            ),
        ]:
            started = time.monotonic()
            result = env.execute({"command": command})
            assert result["returncode"] == -1
            assert "deadline" in result["exception_info"]
            assert time.monotonic() - started < 10
        r_pid = env.execute({"command": "if test -f /tmp/loop.pid; then test ! -d /proc/$(cat /tmp/loop.pid); fi"})
        assert r_pid["returncode"] == 0
    finally:
        cid = env.container_id
        env.cleanup()
    assert env.container_id is None
    chk = subprocess.run([executable, "inspect", cid or "removed"], capture_output=True)
    assert chk.returncode != 0

    mem_env = DockerEnvironment(
        image="python:3.11",
        executable=executable,
        cwd="/tmp",
        timeout=10,
        terminate_grace_seconds=0.2,
        run_args=["--rm", "--init", "--memory", "64m", "--memory-swap", "64m", "--cpus", "0.5", "--pids-limit", "64"],
    )
    try:
        result = mem_env.execute({"command": "python3 -c 'x=[bytearray(16*1024*1024) for _ in range(32)]'"}, timeout=10)
        assert result["returncode"] == -9
        assert "SIGKILL" in result["exception_info"]
    finally:
        mem_env.cleanup()

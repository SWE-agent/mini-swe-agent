import json
import os
import subprocess
import time
from unittest.mock import patch

import pytest

from minisweagent.environments.docker import DockerEnvironment, DockerEnvironmentConfig


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
    assert config.isolate_network


@pytest.mark.slow
def test_default_networks_isolate_concurrent_environments(container_executable):
    envs = [DockerEnvironment(image="python:3.12-slim", executable=container_executable, cwd="/tmp") for _ in range(2)]
    network_names = [env.network_name for env in envs]
    try:
        assert network_names[0] and network_names[1] and network_names[0] != network_names[1]
        inspect = subprocess.run(
            [container_executable, "inspect", envs[0].container_id, envs[1].container_id],
            capture_output=True,
            text=True,
            check=True,
        ).stdout
        container_info = json.loads(inspect)
        assert [set(info["NetworkSettings"]["Networks"]) for info in container_info] == [
            {network_names[0]},
            {network_names[1]},
        ]
        network_info = json.loads(
            subprocess.run(
                [container_executable, "network", "inspect", *network_names],
                capture_output=True,
                text=True,
                check=True,
            ).stdout
        )
        assert all(not info["Internal"] for info in network_info)

        envs[0].execute({"command": "echo private > marker; python -m http.server 18080 >/tmp/http.log 2>&1 &"})
        first_ip = container_info[0]["NetworkSettings"]["Networks"][network_names[0]]["IPAddress"]
        local = None
        for _ in range(30):
            local = envs[0].execute(
                {
                    "command": 'python -c "import urllib.request; '
                    "print(urllib.request.urlopen('http://127.0.0.1:18080/marker').read().decode())\""
                }
            )
            if local["returncode"] == 0:
                break
            time.sleep(0.1)
        assert local and local["output"].strip() == "private"
        probe = envs[1].execute(
            {"command": f"python -c \"import socket; socket.create_connection(('{first_ip}',18080),timeout=2)\""}
        )
        assert probe["returncode"] != 0
        assert envs[1].execute({"command": "test ! -e /tmp/marker"})["returncode"] == 0
    finally:
        for env in envs:
            env.cleanup()
        for _ in range(50):
            if all(
                subprocess.run([container_executable, "network", "inspect", name], capture_output=True).returncode != 0
                for name in network_names
            ):
                break
            time.sleep(0.1)
        assert all(
            subprocess.run([container_executable, "network", "inspect", name], capture_output=True).returncode != 0
            for name in network_names
        )


@pytest.mark.slow
def test_cleanup_preserves_container_without_rm(container_executable):
    env = DockerEnvironment(
        image="python:3.12-slim", executable=container_executable, cwd="/tmp", run_args=[]
    )
    container_id = env.container_id
    network_name = env.network_name
    try:
        env.cleanup()
        inspect = subprocess.run(
            [container_executable, "inspect", container_id], capture_output=True, text=True, check=True
        )
        assert json.loads(inspect.stdout)[0]["State"]["Status"] == "exited"
        assert subprocess.run(
            [container_executable, "network", "inspect", network_name], capture_output=True
        ).returncode == 0
    finally:
        subprocess.run([container_executable, "rm", container_id], capture_output=True)
        subprocess.run([container_executable, "network", "rm", network_name], capture_output=True)


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

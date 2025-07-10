import os
import subprocess
from unittest.mock import patch

import pytest

from microswea.environments.docker import DockerEnvironment, DockerEnvironmentConfig
from microswea.environments.podman import PodmanEnvironment, PodmanEnvironmentConfig


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
        DockerEnvironment,
        DockerEnvironmentConfig,
        "docker",
        marks=pytest.mark.skipif(not is_docker_available(), reason="Docker not available"),
        id="docker",
    ),
    pytest.param(
        PodmanEnvironment,
        PodmanEnvironmentConfig,
        "podman",
        marks=pytest.mark.skipif(not is_podman_available(), reason="Podman not available"),
        id="podman",
    ),
]


@pytest.mark.parametrize(("env_class", "config_class", "executable"), environment_params)
def test_docker_environment_config_defaults(env_class, config_class, executable):
    """Test that DockerEnvironmentConfig has correct default values."""
    config = config_class(image="python:3.11")

    assert config.image == "python:3.11"
    assert config.cwd == "/"
    assert config.env == {}
    assert config.forward_env == []
    assert config.timeout == 30
    assert config.executable == executable


@pytest.mark.slow
@pytest.mark.parametrize(("env_class", "config_class", "executable"), environment_params)
def test_docker_environment_basic_execution(env_class, config_class, executable):
    """Test basic command execution in Docker container."""
    env = env_class(image="python:3.11")

    try:
        result = env.execute("echo 'hello world'")
        assert result["returncode"] == 0
        assert "hello world" in result["stdout"]
    finally:
        env.cleanup()


@pytest.mark.slow
@pytest.mark.parametrize(("env_class", "config_class", "executable"), environment_params)
def test_docker_environment_set_env_variables(env_class, config_class, executable):
    """Test setting environment variables in the container."""
    env = env_class(image="python:3.11", env={"TEST_VAR": "test_value", "ANOTHER_VAR": "another_value"})

    try:
        # Test single environment variable
        result = env.execute("echo $TEST_VAR")
        assert result["returncode"] == 0
        assert "test_value" in result["stdout"]

        # Test multiple environment variables
        result = env.execute("echo $TEST_VAR $ANOTHER_VAR")
        assert result["returncode"] == 0
        assert "test_value another_value" in result["stdout"]
    finally:
        env.cleanup()


@pytest.mark.slow
@pytest.mark.parametrize(("env_class", "config_class", "executable"), environment_params)
def test_docker_environment_forward_env_variables(env_class, config_class, executable):
    """Test forwarding environment variables from host to container."""
    with patch.dict(os.environ, {"HOST_VAR": "host_value", "ANOTHER_HOST_VAR": "another_host_value"}):
        env = env_class(image="python:3.11", forward_env=["HOST_VAR", "ANOTHER_HOST_VAR"])

        try:
            # Test single forwarded environment variable
            result = env.execute("echo $HOST_VAR")
            assert result["returncode"] == 0
            assert "host_value" in result["stdout"]

            # Test multiple forwarded environment variables
            result = env.execute("echo $HOST_VAR $ANOTHER_HOST_VAR")
            assert result["returncode"] == 0
            assert "host_value another_host_value" in result["stdout"]
        finally:
            env.cleanup()


@pytest.mark.slow
@pytest.mark.parametrize(("env_class", "config_class", "executable"), environment_params)
def test_docker_environment_forward_nonexistent_env_variables(env_class, config_class, executable):
    """Test forwarding non-existent environment variables (should be empty)."""
    env = env_class(image="python:3.11", forward_env=["NONEXISTENT_VAR"])

    try:
        result = env.execute('echo "[$NONEXISTENT_VAR]"')
        assert result["returncode"] == 0
        assert "[]" in result["stdout"]  # Empty variable should result in empty string
    finally:
        env.cleanup()


@pytest.mark.slow
@pytest.mark.parametrize(("env_class", "config_class", "executable"), environment_params)
def test_docker_environment_combined_env_and_forward(env_class, config_class, executable):
    """Test both setting and forwarding environment variables together."""
    with patch.dict(os.environ, {"HOST_VAR": "from_host"}):
        env = env_class(image="python:3.11", env={"SET_VAR": "from_config"}, forward_env=["HOST_VAR"])

        try:
            result = env.execute("echo $SET_VAR $HOST_VAR")
            assert result["returncode"] == 0
            assert "from_config from_host" in result["stdout"]
        finally:
            env.cleanup()


@pytest.mark.slow
@pytest.mark.parametrize(("env_class", "config_class", "executable"), environment_params)
def test_docker_environment_env_override_forward(env_class, config_class, executable):
    """Test that explicitly set env variables take precedence over forwarded ones."""
    with patch.dict(os.environ, {"CONFLICT_VAR": "from_host"}):
        env = env_class(image="python:3.11", env={"CONFLICT_VAR": "from_config"}, forward_env=["CONFLICT_VAR"])

        try:
            result = env.execute("echo $CONFLICT_VAR")
            assert result["returncode"] == 0
            # The explicitly set env should take precedence (comes first in docker exec command)
            assert "from_config" in result["stdout"]
        finally:
            env.cleanup()


@pytest.mark.slow
@pytest.mark.parametrize(("env_class", "config_class", "executable"), environment_params)
def test_docker_environment_custom_cwd(env_class, config_class, executable):
    """Test executing commands in a custom working directory."""
    env = env_class(image="python:3.11", cwd="/tmp")

    try:
        result = env.execute("pwd")
        assert result["returncode"] == 0
        assert "/tmp" in result["stdout"]
    finally:
        env.cleanup()


@pytest.mark.slow
@pytest.mark.parametrize(("env_class", "config_class", "executable"), environment_params)
def test_docker_environment_cwd_parameter_override(env_class, config_class, executable):
    """Test that the cwd parameter in execute() overrides the config cwd."""
    env = env_class(image="python:3.11", cwd="/")

    try:
        result = env.execute("pwd", cwd="/tmp")
        assert result["returncode"] == 0
        assert "/tmp" in result["stdout"]
    finally:
        env.cleanup()


@pytest.mark.slow
@pytest.mark.parametrize(("env_class", "config_class", "executable"), environment_params)
def test_docker_environment_command_failure(env_class, config_class, executable):
    """Test that command failures are properly captured."""
    env = env_class(image="python:3.11")

    try:
        result = env.execute("exit 42")
        assert result["returncode"] == 42
    finally:
        env.cleanup()

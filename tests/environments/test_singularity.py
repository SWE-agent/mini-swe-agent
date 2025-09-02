import os
import subprocess
from unittest.mock import patch

import pytest

from minisweagent.environments.singularity import SingularityEnvironment, SingularityEnvironmentConfig


def is_singularity_available():
    """Check if Singularity is available."""
    try:
        subprocess.run(["singularity", "version"], capture_output=True, check=True, timeout=5)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired):
        return False


@pytest.mark.skipif(not is_singularity_available(), reason="Singularity not available")
def test_singularity_environment_config_defaults():
    """Test that SingularityEnvironmentConfig has correct default values."""
    config = SingularityEnvironmentConfig(image="python:3.11")

    assert config.image == "python:3.11"
    assert config.cwd == "/"
    assert config.env == {}
    assert config.forward_env == []
    assert config.timeout == 30
    assert config.executable == "singularity"


@pytest.mark.slow
@pytest.mark.skipif(not is_singularity_available(), reason="Singularity not available")
def test_singularity_environment_basic_execution():
    """Test basic command execution in Singularity container."""
    # Using a lightweight image that should be available or easily pulled
    env = SingularityEnvironment(image="docker://python:3.11-slim")

    result = env.execute("echo 'hello world'")
    assert result["returncode"] == 0
    assert "hello world" in result["output"]


@pytest.mark.slow
@pytest.mark.skipif(not is_singularity_available(), reason="Singularity not available")
def test_singularity_environment_set_env_variables():
    """Test setting environment variables in the container."""
    env = SingularityEnvironment(
        image="docker://python:3.11-slim", env={"TEST_VAR": "test_value", "ANOTHER_VAR": "another_value"}
    )

    # Test single environment variable
    result = env.execute("echo $TEST_VAR")
    assert result["returncode"] == 0
    assert "test_value" in result["output"]

    # Test multiple environment variables
    result = env.execute("echo $TEST_VAR $ANOTHER_VAR")
    assert result["returncode"] == 0
    assert "test_value another_value" in result["output"]


@pytest.mark.slow
@pytest.mark.skipif(not is_singularity_available(), reason="Singularity not available")
def test_singularity_environment_forward_env_variables():
    """Test forwarding environment variables from host to container."""
    with patch.dict(os.environ, {"HOST_VAR": "host_value", "ANOTHER_HOST_VAR": "another_host_value"}):
        env = SingularityEnvironment(image="docker://python:3.11-slim", forward_env=["HOST_VAR", "ANOTHER_HOST_VAR"])

        # Test single forwarded environment variable
        result = env.execute("echo $HOST_VAR")
        assert result["returncode"] == 0
        assert "host_value" in result["output"]

        # Test multiple forwarded environment variables
        result = env.execute("echo $HOST_VAR $ANOTHER_HOST_VAR")
        assert result["returncode"] == 0
        assert "host_value another_host_value" in result["output"]


@pytest.mark.slow
@pytest.mark.skipif(not is_singularity_available(), reason="Singularity not available")
def test_singularity_environment_forward_nonexistent_env_variables():
    """Test forwarding non-existent environment variables (should be empty)."""
    env = SingularityEnvironment(image="docker://python:3.11-slim", forward_env=["NONEXISTENT_VAR"])

    result = env.execute('echo "[$NONEXISTENT_VAR]"')
    assert result["returncode"] == 0
    assert "[]" in result["output"]  # Empty variable should result in empty string


@pytest.mark.slow
@pytest.mark.skipif(not is_singularity_available(), reason="Singularity not available")
def test_singularity_environment_combined_env_and_forward():
    """Test both setting and forwarding environment variables together."""
    with patch.dict(os.environ, {"HOST_VAR": "from_host"}):
        env = SingularityEnvironment(
            image="docker://python:3.11-slim", env={"SET_VAR": "from_config"}, forward_env=["HOST_VAR"]
        )

        result = env.execute("echo $SET_VAR $HOST_VAR")
        assert result["returncode"] == 0
        assert "from_config from_host" in result["output"]


@pytest.mark.slow
@pytest.mark.skipif(not is_singularity_available(), reason="Singularity not available")
def test_singularity_environment_env_override_forward():
    """Test that explicitly set env variables take precedence over forwarded ones."""
    with patch.dict(os.environ, {"CONFLICT_VAR": "from_host"}):
        env = SingularityEnvironment(
            image="docker://python:3.11-slim", env={"CONFLICT_VAR": "from_config"}, forward_env=["CONFLICT_VAR"]
        )

        result = env.execute("echo $CONFLICT_VAR")
        assert result["returncode"] == 0
        # The explicitly set env should take precedence (comes after forwarded in singularity exec command)
        assert "from_config" in result["output"]


@pytest.mark.slow
@pytest.mark.skipif(not is_singularity_available(), reason="Singularity not available")
def test_singularity_environment_custom_cwd():
    """Test executing commands in a custom working directory."""
    env = SingularityEnvironment(image="docker://python:3.11-slim", cwd="/tmp")

    result = env.execute("pwd")
    assert result["returncode"] == 0
    assert "/tmp" in result["output"]


@pytest.mark.slow
@pytest.mark.skipif(not is_singularity_available(), reason="Singularity not available")
def test_singularity_environment_cwd_parameter_override():
    """Test that the cwd parameter in execute() overrides the config cwd."""
    env = SingularityEnvironment(image="docker://python:3.11-slim", cwd="/")

    result = env.execute("pwd", cwd="/tmp")
    assert result["returncode"] == 0
    assert "/tmp" in result["output"]


@pytest.mark.slow
@pytest.mark.skipif(not is_singularity_available(), reason="Singularity not available")
def test_singularity_environment_command_failure():
    """Test that command failures are properly captured."""
    env = SingularityEnvironment(image="docker://python:3.11-slim")

    result = env.execute("exit 42")
    assert result["returncode"] == 42


@pytest.mark.slow
@pytest.mark.skipif(not is_singularity_available(), reason="Singularity not available")
def test_singularity_environment_timeout():
    """Test that the timeout configuration is respected."""
    env = SingularityEnvironment(image="docker://python:3.11-slim", timeout=1)

    # This should timeout and raise TimeoutExpired
    with pytest.raises(subprocess.TimeoutExpired):
        env.execute("sleep 5")

import os
import subprocess
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

from minisweagent.environments.local import LocalEnvironment, LocalEnvironmentConfig


def test_local_environment_config_defaults():
    """Test that LocalEnvironmentConfig has correct default values."""
    config = LocalEnvironmentConfig()

    assert config.cwd == ""
    assert config.env == {}
    assert config.timeout == 30


def test_local_environment_basic_execution():
    """Test basic command execution in local environment."""
    env = LocalEnvironment()

    result = env.execute("echo 'hello world'")
    assert result["returncode"] == 0
    assert "hello world" in result["output"]


def test_local_environment_set_env_variables():
    """Test setting environment variables in the local environment."""
    env = LocalEnvironment(env={"TEST_VAR": "test_value", "ANOTHER_VAR": "another_value"})

    # Test single environment variable
    result = env.execute("echo $TEST_VAR")
    assert result["returncode"] == 0
    assert "test_value" in result["output"]

    # Test multiple environment variables
    result = env.execute("echo $TEST_VAR $ANOTHER_VAR")
    assert result["returncode"] == 0
    assert "test_value another_value" in result["output"]


def test_local_environment_existing_env_variables():
    """Test that existing environment variables are preserved and merged."""
    with patch.dict(os.environ, {"EXISTING_VAR": "existing_value"}):
        env = LocalEnvironment(env={"NEW_VAR": "new_value"})

        # Test that both existing and new variables are available
        result = env.execute("echo $EXISTING_VAR $NEW_VAR")
        assert result["returncode"] == 0
        assert "existing_value new_value" in result["output"]


def test_local_environment_env_variable_override():
    """Test that config env variables override existing ones."""
    with patch.dict(os.environ, {"CONFLICT_VAR": "original_value"}):
        env = LocalEnvironment(env={"CONFLICT_VAR": "override_value"})

        result = env.execute("echo $CONFLICT_VAR")
        assert result["returncode"] == 0
        assert "override_value" in result["output"]


def test_local_environment_custom_cwd():
    """Test executing commands in a custom working directory."""
    with tempfile.TemporaryDirectory() as temp_dir:
        env = LocalEnvironment(cwd=temp_dir)

        result = env.execute("pwd")
        assert result["returncode"] == 0
        assert temp_dir in result["output"]


def test_local_environment_cwd_parameter_override():
    """Test that the cwd parameter in execute() overrides the config cwd."""
    with tempfile.TemporaryDirectory() as temp_dir1, tempfile.TemporaryDirectory() as temp_dir2:
        env = LocalEnvironment(cwd=temp_dir1)

        # Execute with different cwd parameter
        result = env.execute("pwd", cwd=temp_dir2)
        assert result["returncode"] == 0
        assert temp_dir2 in result["output"]


def test_local_environment_default_cwd():
    """Test that commands use os.getcwd() when no cwd is specified."""
    env = LocalEnvironment()
    current_dir = os.getcwd()

    result = env.execute("pwd")
    assert result["returncode"] == 0
    assert current_dir in result["output"]


def test_local_environment_command_failure():
    """Test that command failures are properly captured."""
    env = LocalEnvironment()

    result = env.execute("exit 1")
    assert result["returncode"] == 1
    assert result["output"] == ""


def test_local_environment_nonexistent_command():
    """Test execution of non-existent command."""
    env = LocalEnvironment()

    result = env.execute("nonexistent_command_12345")
    assert result["returncode"] != 0
    assert "nonexistent_command_12345" in result["output"] or "command not found" in result["output"]


def test_local_environment_stderr_capture():
    """Test that stderr is properly captured."""
    env = LocalEnvironment()

    result = env.execute("echo 'error message' >&2")
    assert result["returncode"] == 0
    assert "error message" in result["output"]


def test_local_environment_timeout():
    """Test timeout functionality."""
    env = LocalEnvironment(timeout=1)

    with pytest.raises(subprocess.TimeoutExpired):
        env.execute("sleep 2")


def test_local_environment_custom_timeout():
    """Test custom timeout configuration."""
    config = LocalEnvironmentConfig(timeout=5)
    env = LocalEnvironment(**config.__dict__)

    assert env.config.timeout == 5


@pytest.mark.parametrize(
    ("command", "expected_returncode"),
    [
        ("echo 'test'", 0),
        ("exit 1", 1),
        ("exit 42", 42),
    ],
)
def test_local_environment_return_codes(command, expected_returncode):
    """Test that various return codes are properly captured."""
    env = LocalEnvironment()

    result = env.execute(command)
    assert result["returncode"] == expected_returncode


def test_local_environment_multiline_output():
    """Test handling of multiline command output."""
    env = LocalEnvironment()

    result = env.execute("echo -e 'line1\\nline2\\nline3'")
    assert result["returncode"] == 0
    output_lines = result["output"].strip().split("\n")
    assert len(output_lines) == 3
    assert "line1" in output_lines[0]
    assert "line2" in output_lines[1]
    assert "line3" in output_lines[2]


def test_local_environment_file_operations():
    """Test file operations in the local environment."""
    with tempfile.TemporaryDirectory() as temp_dir:
        env = LocalEnvironment(cwd=temp_dir)

        # Create a file
        result = env.execute("echo 'test content' > test.txt")
        assert result["returncode"] == 0

        # Read the file
        result = env.execute("cat test.txt")
        assert result["returncode"] == 0
        assert "test content" in result["output"]

        # Verify file exists
        test_file = Path(temp_dir) / "test.txt"
        assert test_file.exists()
        assert test_file.read_text().strip() == "test content"


def test_local_environment_shell_features():
    """Test that shell features like pipes and redirects work."""
    env = LocalEnvironment()

    # Test pipe
    result = env.execute("echo 'hello world' | grep 'world'")
    assert result["returncode"] == 0
    assert "hello world" in result["output"]

    # Test command substitution
    result = env.execute("echo $(echo 'nested')")
    assert result["returncode"] == 0
    assert "nested" in result["output"]

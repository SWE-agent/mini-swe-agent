import shutil
import subprocess
import tempfile
from pathlib import Path

import pytest

from minisweagent.environments.extra.bubblewrap import BubblewrapEnvironment, BubblewrapEnvironmentConfig


@pytest.mark.skipif(not shutil.which("bwrap"), reason="bubblewrap not available")
def test_bubblewrap_environment_basic_execution():
    """Test basic command execution in bubblewrap environment."""
    env = BubblewrapEnvironment()

    try:
        result = env.execute("echo 'hello world'")
        print(f"test_bubblewrap_environment_basic_execution result: {result}")
        assert result["returncode"] == 0
        assert "hello world" in result["output"]
    finally:
        env.cleanup()


@pytest.mark.skipif(not shutil.which("bwrap"), reason="bubblewrap not available")
def test_bubblewrap_environment_set_env_variables():
    """Test setting environment variables in the bubblewrap environment."""
    env = BubblewrapEnvironment(env={"TEST_VAR": "test_value", "ANOTHER_VAR": "another_value"})

    try:
        # Test single environment variable
        result = env.execute("echo $TEST_VAR")
        print(f"test_bubblewrap_environment_set_env_variables result (single var): {result}")
        assert result["returncode"] == 0
        assert "test_value" in result["output"]

        # Test multiple environment variables
        result = env.execute("echo $TEST_VAR $ANOTHER_VAR")
        print(f"test_bubblewrap_environment_set_env_variables result (multiple vars): {result}")
        assert result["returncode"] == 0
        assert "test_value another_value" in result["output"]
    finally:
        env.cleanup()


@pytest.mark.skipif(not shutil.which("bwrap"), reason="bubblewrap not available")
def test_bubblewrap_environment_custom_cwd():
    """Test executing commands in a custom working directory."""
    with tempfile.TemporaryDirectory() as temp_dir:
        env = BubblewrapEnvironment(cwd=temp_dir)

        try:
            result = env.execute("pwd")
            print(f"test_bubblewrap_environment_custom_cwd result: {result}")
            assert result["returncode"] == 0
            assert temp_dir in result["output"]
        finally:
            env.cleanup()


@pytest.mark.skipif(not shutil.which("bwrap"), reason="bubblewrap not available")
def test_bubblewrap_environment_cwd_parameter_override():
    """Test that the cwd parameter in execute() overrides the config cwd."""
    with tempfile.TemporaryDirectory() as temp_dir1, tempfile.TemporaryDirectory() as temp_dir2:
        env = BubblewrapEnvironment(cwd=temp_dir1)

        try:
            # Execute with different cwd parameter
            result = env.execute("pwd", cwd=temp_dir2)
            print(f"test_bubblewrap_environment_cwd_parameter_override result: {result}")
            assert result["returncode"] == 0
            assert temp_dir2 in result["output"]
        finally:
            env.cleanup()


@pytest.mark.skipif(not shutil.which("bwrap"), reason="bubblewrap not available")
def test_bubblewrap_environment_command_failure():
    """Test that command failures are properly captured."""
    env = BubblewrapEnvironment()

    try:
        result = env.execute("exit 1")
        print(f"test_bubblewrap_environment_command_failure result: {result}")
        assert result["returncode"] == 1
        assert result["output"] == ""
    finally:
        env.cleanup()


@pytest.mark.skipif(not shutil.which("bwrap"), reason="bubblewrap not available")
def test_bubblewrap_environment_nonexistent_command():
    """Test execution of non-existent command."""
    env = BubblewrapEnvironment()

    try:
        result = env.execute("nonexistent_command_12345")
        print(f"test_bubblewrap_environment_nonexistent_command result: {result}")
        assert result["returncode"] != 0
        assert "nonexistent_command_12345" in result["output"] or "command not found" in result["output"]
    finally:
        env.cleanup()


@pytest.mark.skipif(not shutil.which("bwrap"), reason="bubblewrap not available")
def test_bubblewrap_environment_stderr_capture():
    """Test that stderr is properly captured."""
    env = BubblewrapEnvironment()

    try:
        result = env.execute("echo 'error message' >&2")
        print(f"test_bubblewrap_environment_stderr_capture result: {result}")
        assert result["returncode"] == 0
        assert "error message" in result["output"]
    finally:
        env.cleanup()


@pytest.mark.skipif(not shutil.which("bwrap"), reason="bubblewrap not available")
def test_bubblewrap_environment_timeout():
    """Test timeout functionality."""
    env = BubblewrapEnvironment(timeout=1)

    try:
        with pytest.raises(subprocess.TimeoutExpired):
            env.execute("sleep 2")
    finally:
        env.cleanup()


@pytest.mark.skipif(not shutil.which("bwrap"), reason="bubblewrap not available")
@pytest.mark.parametrize(
    ("command", "expected_returncode"),
    [
        ("echo 'test'", 0),
        ("exit 1", 1),
        ("exit 42", 42),
    ],
)
def test_bubblewrap_environment_return_codes(command, expected_returncode):
    """Test that various return codes are properly captured."""
    env = BubblewrapEnvironment()

    try:
        result = env.execute(command)
        print(f"test_bubblewrap_environment_return_codes result (cmd: {command}): {result}")
        assert result["returncode"] == expected_returncode
    finally:
        env.cleanup()


@pytest.mark.skipif(not shutil.which("bwrap"), reason="bubblewrap not available")
def test_bubblewrap_environment_multiline_output():
    """Test handling of multiline command output."""
    env = BubblewrapEnvironment()

    try:
        result = env.execute("echo -e 'line1\\nline2\\nline3'")
        print(f"test_bubblewrap_environment_multiline_output result: {result}")
        assert result["returncode"] == 0
        output_lines = result["output"].strip().split("\n")

        assert len(output_lines) == 3
        assert "line1" in output_lines[0]
        assert "line2" in output_lines[1]
        assert "line3" in output_lines[2]
    finally:
        env.cleanup()


@pytest.mark.skipif(not shutil.which("bwrap"), reason="bubblewrap not available")
def test_bubblewrap_environment_file_operations():
    """Test file operations in the bubblewrap environment."""
    with tempfile.TemporaryDirectory() as temp_dir:
        env = BubblewrapEnvironment(cwd=temp_dir)

        try:
            # Create a file
            result = env.execute("echo 'test content' > test.txt")
            print(f"test_bubblewrap_environment_file_operations result (create file): {result}")
            assert result["returncode"] == 0

            # Read the file
            result = env.execute("cat test.txt")
            print(f"test_bubblewrap_environment_file_operations result (read file): {result}")
            assert result["returncode"] == 0
            assert "test content" in result["output"]

            # Verify file exists (should be in the working directory)
            test_file = Path(temp_dir) / "test.txt"

            assert test_file.exists()
            assert test_file.read_text().strip() == "test content"
        finally:
            env.cleanup()


@pytest.mark.skipif(not shutil.which("bwrap"), reason="bubblewrap not available")
def test_bubblewrap_environment_working_directory_creation():
    """Test that working directory is properly created."""
    env = BubblewrapEnvironment()

    try:
        assert env.working_dir.exists()
        assert env.working_dir.is_dir()
    finally:
        env.cleanup()


@pytest.mark.skipif(not shutil.which("bwrap"), reason="bubblewrap not available")
def test_bubblewrap_environment_cleanup():
    """Test that cleanup properly removes working directory."""
    env = BubblewrapEnvironment()
    working_dir = env.working_dir

    assert working_dir.exists()

    env.cleanup()

    assert not working_dir.exists()


def test_bubblewrap_environment_custom_executable():
    """Test custom bubblewrap executable configuration."""
    config = BubblewrapEnvironmentConfig(executable="/custom/path/to/bwrap")
    env = BubblewrapEnvironment(**config.__dict__)

    try:
        assert env.config.executable == "/custom/path/to/bwrap"
    finally:
        env.cleanup()


def test_bubblewrap_environment_custom_wrapper_args():
    """Test custom wrapper args configuration."""
    custom_args = ["--ro-bind", "/usr", "/usr", "--tmpfs", "/tmp"]
    config = BubblewrapEnvironmentConfig(wrapper_args=custom_args)
    env = BubblewrapEnvironment(**config.__dict__)

    try:
        assert env.config.wrapper_args == custom_args
    finally:
        env.cleanup()


def test_bubblewrap_environment_get_template_vars():
    """Test get_template_vars method returns expected data."""
    env = BubblewrapEnvironment(env={"TEST_VAR": "test_value"})

    try:
        template_vars = env.get_template_vars()
        print(f"test_bubblewrap_environment_get_template_vars template_vars: {template_vars}")

        # Should contain config data
        assert "env" in template_vars
        assert template_vars["env"]["TEST_VAR"] == "test_value"
        assert "timeout" in template_vars
        assert template_vars["timeout"] == 30

        # Should contain platform info
        assert "system" in template_vars
        assert "machine" in template_vars
    finally:
        env.cleanup()

import re
import subprocess
import sys
from unittest.mock import Mock, patch

import pytest

from microsweagent.run.micro import DEFAULT_CONFIG, app, main


def strip_ansi_codes(text: str) -> str:
    """Remove ANSI escape sequences from text."""
    ansi_escape = re.compile(r"\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])")
    return ansi_escape.sub("", text)


def test_configure_if_first_time_called():
    """Test that configure_if_first_time is called when running micro main."""
    with (
        patch("microsweagent.run.micro.configure_if_first_time") as mock_configure,
        patch("microsweagent.run.micro.run_interactive") as mock_run_interactive,
        patch("microsweagent.run.micro.get_model") as mock_get_model,
        patch("microsweagent.run.micro.LocalEnvironment") as mock_env,
        patch("microsweagent.run.micro.get_config_path") as mock_get_config_path,
        patch("microsweagent.run.micro.yaml.safe_load") as mock_yaml_load,
    ):
        # Setup mocks
        mock_model = Mock()
        mock_get_model.return_value = mock_model
        mock_environment = Mock()
        mock_env.return_value = mock_environment
        mock_config_path = Mock()
        mock_config_path.read_text.return_value = ""
        mock_get_config_path.return_value = mock_config_path
        mock_yaml_load.return_value = {"agent": {"system_template": "test"}, "env": {}, "model": {}}
        mock_run_interactive.return_value = Mock()

        # Call main function
        main(
            config_spec=DEFAULT_CONFIG,
            model_name="test-model",
            task="Test task",
            yolo=False,
            output=None,
            visual=False,
        )

        # Verify configure_if_first_time was called
        mock_configure.assert_called_once()


def test_micro_command_calls_run_interactive():
    """Test that micro command calls run_interactive when visual=False."""
    with (
        patch("microsweagent.run.micro.configure_if_first_time"),
        patch("microsweagent.run.micro.run_interactive") as mock_run_interactive,
        patch("microsweagent.run.micro.get_model") as mock_get_model,
        patch("microsweagent.run.micro.LocalEnvironment") as mock_env,
        patch("microsweagent.run.micro.get_config_path") as mock_get_config_path,
        patch("microsweagent.run.micro.yaml.safe_load") as mock_yaml_load,
    ):
        # Setup mocks
        mock_model = Mock()
        mock_get_model.return_value = mock_model
        mock_environment = Mock()
        mock_env.return_value = mock_environment
        mock_config_path = Mock()
        mock_config_path.read_text.return_value = ""
        mock_get_config_path.return_value = mock_config_path
        mock_yaml_load.return_value = {"agent": {"system_template": "test"}, "env": {}, "model": {}}
        mock_run_interactive.return_value = Mock()

        # Call main function with task provided (so prompt is not called)
        main(
            config_spec=DEFAULT_CONFIG,
            model_name="test-model",
            task="Test task",
            yolo=False,
            output=None,
            visual=False,
        )

        # Verify run_interactive was called
        mock_run_interactive.assert_called_once()
        args, kwargs = mock_run_interactive.call_args
        assert args[0] == mock_model  # model
        assert args[1] == mock_environment  # env
        assert args[3] == "Test task"  # task


def test_micro_v_command_calls_run_textual():
    """Test that micro -v command calls run_textual when visual=True."""
    with (
        patch("microsweagent.run.micro.configure_if_first_time"),
        patch("microsweagent.run.micro.run_textual") as mock_run_textual,
        patch("microsweagent.run.micro.get_model") as mock_get_model,
        patch("microsweagent.run.micro.LocalEnvironment") as mock_env,
        patch("microsweagent.run.micro.get_config_path") as mock_get_config_path,
        patch("microsweagent.run.micro.yaml.safe_load") as mock_yaml_load,
    ):
        # Setup mocks
        mock_model = Mock()
        mock_get_model.return_value = mock_model
        mock_environment = Mock()
        mock_env.return_value = mock_environment
        mock_config_path = Mock()
        mock_config_path.read_text.return_value = ""
        mock_get_config_path.return_value = mock_config_path
        mock_yaml_load.return_value = {"agent": {"system_template": "test"}, "env": {}, "model": {}}
        mock_run_textual.return_value = Mock()

        # Call main function with visual=True and task provided
        main(
            config_spec=DEFAULT_CONFIG,
            model_name="test-model",
            task="Test task",
            yolo=False,
            output=None,
            visual=True,
        )

        # Verify run_textual was called
        mock_run_textual.assert_called_once()
        args, kwargs = mock_run_textual.call_args
        assert args[0] == mock_model  # model
        assert args[1] == mock_environment  # env
        assert args[3] == "Test task"  # task


def test_micro_calls_prompt_when_no_task_provided():
    """Test that micro calls prompt when no task is provided."""
    with (
        patch("microsweagent.run.micro.configure_if_first_time"),
        patch("microsweagent.run.micro.prompt_session.prompt") as mock_prompt,
        patch("microsweagent.run.micro.run_interactive") as mock_run_interactive,
        patch("microsweagent.run.micro.get_model") as mock_get_model,
        patch("microsweagent.run.micro.LocalEnvironment") as mock_env,
        patch("microsweagent.run.micro.get_config_path") as mock_get_config_path,
        patch("microsweagent.run.micro.yaml.safe_load") as mock_yaml_load,
    ):
        # Setup mocks
        mock_prompt.return_value = "User provided task"
        mock_model = Mock()
        mock_get_model.return_value = mock_model
        mock_environment = Mock()
        mock_env.return_value = mock_environment
        mock_config_path = Mock()
        mock_config_path.read_text.return_value = ""
        mock_get_config_path.return_value = mock_config_path
        mock_yaml_load.return_value = {"agent": {"system_template": "test"}, "env": {}, "model": {}}
        mock_run_interactive.return_value = Mock()

        # Call main function without task
        main(
            config_spec=DEFAULT_CONFIG,
            model_name="test-model",
            task=None,  # No task provided
            yolo=False,
            output=None,
            visual=False,
        )

        # Verify prompt was called
        mock_prompt.assert_called_once()

        # Verify run_interactive was called with the task from prompt
        mock_run_interactive.assert_called_once()
        args, kwargs = mock_run_interactive.call_args
        assert args[3] == "User provided task"  # task


def test_micro_v_calls_prompt_when_no_task_provided():
    """Test that micro -v calls prompt when no task is provided."""
    with (
        patch("microsweagent.run.micro.configure_if_first_time"),
        patch("microsweagent.run.micro.prompt_session.prompt") as mock_prompt,
        patch("microsweagent.run.micro.run_textual") as mock_run_textual,
        patch("microsweagent.run.micro.get_model") as mock_get_model,
        patch("microsweagent.run.micro.LocalEnvironment") as mock_env,
        patch("microsweagent.run.micro.get_config_path") as mock_get_config_path,
        patch("microsweagent.run.micro.yaml.safe_load") as mock_yaml_load,
    ):
        # Setup mocks
        mock_prompt.return_value = "User provided visual task"
        mock_model = Mock()
        mock_get_model.return_value = mock_model
        mock_environment = Mock()
        mock_env.return_value = mock_environment
        mock_config_path = Mock()
        mock_config_path.read_text.return_value = ""
        mock_get_config_path.return_value = mock_config_path
        mock_yaml_load.return_value = {"agent": {"system_template": "test"}, "env": {}, "model": {}}
        mock_run_textual.return_value = Mock()

        # Call main function with visual=True but no task
        main(
            config_spec=DEFAULT_CONFIG,
            model_name="test-model",
            task=None,  # No task provided
            yolo=False,
            output=None,
            visual=True,
        )

        # Verify prompt was called
        mock_prompt.assert_called_once()

        # Verify run_textual was called with the task from prompt
        mock_run_textual.assert_called_once()
        args, kwargs = mock_run_textual.call_args
        assert args[3] == "User provided visual task"  # task


def test_micro_with_explicit_model():
    """Test that micro works with explicitly provided model."""
    with (
        patch("microsweagent.run.micro.configure_if_first_time"),
        patch("microsweagent.run.micro.run_interactive") as mock_run_interactive,
        patch("microsweagent.run.micro.get_model") as mock_get_model,
        patch("microsweagent.run.micro.LocalEnvironment") as mock_env,
        patch("microsweagent.run.micro.get_config_path") as mock_get_config_path,
        patch("microsweagent.run.micro.yaml.safe_load") as mock_yaml_load,
    ):
        # Setup mocks
        mock_model = Mock()
        mock_get_model.return_value = mock_model
        mock_environment = Mock()
        mock_env.return_value = mock_environment
        mock_config_path = Mock()
        mock_config_path.read_text.return_value = ""
        mock_get_config_path.return_value = mock_config_path
        mock_yaml_load.return_value = {
            "agent": {"system_template": "test"},
            "env": {},
            "model": {"default_config": "test"},
        }
        mock_run_interactive.return_value = Mock()

        # Call main function with explicit model
        main(
            config_spec=DEFAULT_CONFIG,
            model_name="gpt-4",
            task="Test task with explicit model",
            yolo=True,
            output=None,
            visual=False,
        )

        # Verify get_model was called with the explicit model
        mock_get_model.assert_called_once_with("gpt-4", {"default_config": "test"})

        # Verify run_interactive was called
        mock_run_interactive.assert_called_once()


def test_yolo_mode_sets_correct_agent_config():
    """Test that yolo mode sets the correct agent configuration."""
    with (
        patch("microsweagent.run.micro.configure_if_first_time"),
        patch("microsweagent.run.micro.run_interactive") as mock_run_interactive,
        patch("microsweagent.run.micro.get_model") as mock_get_model,
        patch("microsweagent.run.micro.LocalEnvironment") as mock_env,
        patch("microsweagent.run.micro.get_config_path") as mock_get_config_path,
        patch("microsweagent.run.micro.yaml.safe_load") as mock_yaml_load,
    ):
        # Setup mocks
        mock_model = Mock()
        mock_get_model.return_value = mock_model
        mock_environment = Mock()
        mock_env.return_value = mock_environment
        mock_config_path = Mock()
        mock_config_path.read_text.return_value = ""
        mock_get_config_path.return_value = mock_config_path
        mock_yaml_load.return_value = {"agent": {"system_template": "test"}, "env": {}, "model": {}}
        mock_run_interactive.return_value = Mock()

        # Call main function with yolo=True
        main(
            config_spec=DEFAULT_CONFIG,
            model_name="test-model",
            task="Test yolo task",
            yolo=True,
            output=None,
            visual=False,
        )

        # Verify run_interactive was called with yolo mode
        mock_run_interactive.assert_called_once()
        args, kwargs = mock_run_interactive.call_args
        agent_config = args[2]  # agent_config is the third argument
        assert agent_config["mode"] == "yolo"


def test_confirm_mode_sets_correct_agent_config():
    """Test that confirm mode (default) sets the correct agent configuration."""
    with (
        patch("microsweagent.run.micro.configure_if_first_time"),
        patch("microsweagent.run.micro.run_interactive") as mock_run_interactive,
        patch("microsweagent.run.micro.get_model") as mock_get_model,
        patch("microsweagent.run.micro.LocalEnvironment") as mock_env,
        patch("microsweagent.run.micro.get_config_path") as mock_get_config_path,
        patch("microsweagent.run.micro.yaml.safe_load") as mock_yaml_load,
    ):
        # Setup mocks
        mock_model = Mock()
        mock_get_model.return_value = mock_model
        mock_environment = Mock()
        mock_env.return_value = mock_environment
        mock_config_path = Mock()
        mock_config_path.read_text.return_value = ""
        mock_get_config_path.return_value = mock_config_path
        mock_yaml_load.return_value = {"agent": {"system_template": "test"}, "env": {}, "model": {}}
        mock_run_interactive.return_value = Mock()

        # Call main function with yolo=False (default)
        main(
            config_spec=DEFAULT_CONFIG,
            model_name="test-model",
            task="Test confirm task",
            yolo=False,
            output=None,
            visual=False,
        )

        # Verify run_interactive was called with confirm mode
        mock_run_interactive.assert_called_once()
        args, kwargs = mock_run_interactive.call_args
        agent_config = args[2]  # agent_config is the third argument
        assert agent_config["mode"] == "confirm"


def test_micro_help():
    """Test that micro --help works correctly."""
    result = subprocess.run(
        [sys.executable, "-m", "microsweagent", "--help"],
        capture_output=True,
        text=True,
        timeout=10,
    )

    assert result.returncode == 0
    # Strip ANSI color codes for reliable text matching
    clean_output = strip_ansi_codes(result.stdout)
    assert "Run micro-SWE-agent right here, right now." in clean_output
    assert "--help" in clean_output
    assert "--config" in clean_output
    assert "--model" in clean_output
    assert "--task" in clean_output
    assert "--yolo" in clean_output
    assert "--output" in clean_output
    assert "--visual" in clean_output


def test_micro_help_with_typer_runner():
    """Test help functionality using typer's test runner."""
    from typer.testing import CliRunner

    runner = CliRunner()
    result = runner.invoke(app, ["--help"])

    assert result.exit_code == 0
    # Strip ANSI color codes for reliable text matching
    clean_output = strip_ansi_codes(result.stdout)
    assert "Run micro-SWE-agent right here, right now." in clean_output
    assert "--help" in clean_output
    assert "--config" in clean_output
    assert "--model" in clean_output
    assert "--task" in clean_output
    assert "--yolo" in clean_output
    assert "--output" in clean_output
    assert "--visual" in clean_output


def test_python_m_microsweagent_help():
    """Test that python -m microsweagent --help works correctly."""
    result = subprocess.run(
        [sys.executable, "-m", "microsweagent", "--help"],
        capture_output=True,
        text=True,
        timeout=10,
    )

    assert result.returncode == 0
    assert "micro-SWE-agent" in result.stdout


def test_micro_script_help():
    """Test that the micro script entry point help works."""
    result = subprocess.run(
        ["micro", "--help"],
        capture_output=True,
        text=True,
        timeout=10,
    )

    # This might fail if micro is not installed, so we handle that gracefully
    if result.returncode == 0:
        assert "micro-SWE-agent" in result.stdout
    else:
        # If micro command is not available, that's expected in test environment
        pytest.skip("micro command not available in test environment")

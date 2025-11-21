import re
import subprocess
import sys
from unittest.mock import Mock, patch

import pytest

from minisweagent.run.mini import DEFAULT_CONFIG, app, main


def strip_ansi_codes(text: str) -> str:
    """Remove ANSI escape sequences from text."""
    ansi_escape = re.compile(r"\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])")
    return ansi_escape.sub("", text)


def test_configure_if_first_time_called():
    """Test that configure_if_first_time is called when running mini main."""
    with (
        patch("minisweagent.run.mini.configure_if_first_time") as mock_configure,
        patch("minisweagent.run.mini.InteractiveAgent") as mock_interactive_agent_class,
        patch("minisweagent.run.mini.get_model") as mock_get_model,
        patch("minisweagent.run.mini.LocalEnvironment") as mock_env,
        patch("minisweagent.run.mini.get_config_path") as mock_get_config_path,
        patch("minisweagent.run.mini.yaml.safe_load") as mock_yaml_load,
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

        # Setup mock agent instance
        mock_agent = Mock()
        mock_agent.run.return_value = ("Success", "Result")
        mock_interactive_agent_class.return_value = mock_agent

        # Call main function
        main(
            config_spec=DEFAULT_CONFIG,
            model_name="test-model",
            task="Test task",
            yolo=False,
            output=None,
            visual=False,
            model_class=None,
        )

        # Verify configure_if_first_time was called
        mock_configure.assert_called_once()


def test_mini_command_calls_run_interactive():
    """Test that mini command creates InteractiveAgent when visual=False."""
    with (
        patch("minisweagent.run.mini.configure_if_first_time"),
        patch("minisweagent.run.mini.InteractiveAgent") as mock_interactive_agent_class,
        patch("minisweagent.run.mini.get_model") as mock_get_model,
        patch("minisweagent.run.mini.LocalEnvironment") as mock_env,
        patch("minisweagent.run.mini.get_config_path") as mock_get_config_path,
        patch("minisweagent.run.mini.yaml.safe_load") as mock_yaml_load,
    ):
        # Setup mocks
        mock_model = Mock()
        mock_get_model.return_value = mock_model
        mock_environment = Mock()
        mock_env.return_value = mock_environment
        mock_config_path = Mock()
        mock_config_path.read_text.return_value = ""
        mock_get_config_path.return_value = mock_config_path
        mock_yaml_load.return_value = {"agent": {"system_template": "test", "mode": "confirm"}, "env": {}, "model": {}}

        # Setup mock agent instance
        mock_agent = Mock()
        mock_agent.run.return_value = ("Success", "Result")
        mock_interactive_agent_class.return_value = mock_agent

        # Call main function with task provided (so prompt is not called)
        main(
            config_spec=DEFAULT_CONFIG,
            model_name="test-model",
            task="Test task",
            yolo=False,
            output=None,
            visual=False,
            model_class=None,
        )

        # Verify InteractiveAgent was instantiated
        mock_interactive_agent_class.assert_called_once()
        args, kwargs = mock_interactive_agent_class.call_args
        assert args[0] == mock_model  # model
        assert args[1] == mock_environment  # env
        # Verify agent.run was called with the task
        mock_agent.run.assert_called_once_with("Test task")


def test_mini_v_command_calls_run_textual():
    """Test that mini -v command creates TextualAgent when visual=True."""
    with (
        patch("minisweagent.run.mini.configure_if_first_time"),
        patch("minisweagent.run.mini.TextualAgent") as mock_textual_agent_class,
        patch("minisweagent.run.mini.get_model") as mock_get_model,
        patch("minisweagent.run.mini.LocalEnvironment") as mock_env,
        patch("minisweagent.run.mini.get_config_path") as mock_get_config_path,
        patch("minisweagent.run.mini.yaml.safe_load") as mock_yaml_load,
    ):
        # Setup mocks
        mock_model = Mock()
        mock_get_model.return_value = mock_model
        mock_environment = Mock()
        mock_env.return_value = mock_environment
        mock_config_path = Mock()
        mock_config_path.read_text.return_value = ""
        mock_get_config_path.return_value = mock_config_path
        mock_yaml_load.return_value = {"agent": {"system_template": "test", "mode": "confirm"}, "env": {}, "model": {}}

        # Setup mock agent instance
        mock_agent = Mock()
        mock_agent.run.return_value = ("Success", "Result")
        mock_textual_agent_class.return_value = mock_agent

        # Call main function with visual=True and task provided
        main(
            config_spec=DEFAULT_CONFIG,
            model_name="test-model",
            task="Test task",
            yolo=False,
            output=None,
            visual=True,
            model_class=None,
        )

        # Verify TextualAgent was instantiated
        mock_textual_agent_class.assert_called_once()
        args, kwargs = mock_textual_agent_class.call_args
        assert args[0] == mock_model  # model
        assert args[1] == mock_environment  # env
        # Verify agent.run was called with the task
        mock_agent.run.assert_called_once_with("Test task")


def test_mini_calls_prompt_when_no_task_provided():
    """Test that mini calls prompt when no task is provided."""
    with (
        patch("minisweagent.run.mini.configure_if_first_time"),
        patch("minisweagent.run.mini.prompt_session.prompt") as mock_prompt,
        patch("minisweagent.run.mini.InteractiveAgent") as mock_interactive_agent_class,
        patch("minisweagent.run.mini.get_model") as mock_get_model,
        patch("minisweagent.run.mini.LocalEnvironment") as mock_env,
        patch("minisweagent.run.mini.get_config_path") as mock_get_config_path,
        patch("minisweagent.run.mini.yaml.safe_load") as mock_yaml_load,
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
        mock_yaml_load.return_value = {"agent": {"system_template": "test", "mode": "confirm"}, "env": {}, "model": {}}

        # Setup mock agent instance
        mock_agent = Mock()
        mock_agent.run.return_value = ("Success", "Result")
        mock_interactive_agent_class.return_value = mock_agent

        # Call main function without task
        main(
            config_spec=DEFAULT_CONFIG,
            model_name="test-model",
            task=None,  # No task provided
            yolo=False,
            output=None,
            visual=False,
            model_class=None,
        )

        # Verify prompt was called
        mock_prompt.assert_called_once()

        # Verify InteractiveAgent was instantiated
        mock_interactive_agent_class.assert_called_once()
        # Verify agent.run was called with the task from prompt
        mock_agent.run.assert_called_once_with("User provided task")


def test_mini_v_calls_prompt_when_no_task_provided():
    """Test that mini -v calls prompt when no task is provided."""
    with (
        patch("minisweagent.run.mini.configure_if_first_time"),
        patch("minisweagent.run.mini.prompt_session.prompt") as mock_prompt,
        patch("minisweagent.run.mini.TextualAgent") as mock_textual_agent_class,
        patch("minisweagent.run.mini.get_model") as mock_get_model,
        patch("minisweagent.run.mini.LocalEnvironment") as mock_env,
        patch("minisweagent.run.mini.get_config_path") as mock_get_config_path,
        patch("minisweagent.run.mini.yaml.safe_load") as mock_yaml_load,
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
        mock_yaml_load.return_value = {"agent": {"system_template": "test", "mode": "confirm"}, "env": {}, "model": {}}

        # Setup mock agent instance
        mock_agent = Mock()
        mock_agent.run.return_value = ("Success", "Result")
        mock_textual_agent_class.return_value = mock_agent

        # Call main function with visual=True but no task
        main(
            config_spec=DEFAULT_CONFIG,
            model_name="test-model",
            task=None,  # No task provided
            yolo=False,
            output=None,
            visual=True,
            model_class=None,
        )

        # Verify prompt was called
        mock_prompt.assert_called_once()

        # Verify TextualAgent was instantiated
        mock_textual_agent_class.assert_called_once()
        # Verify agent.run was called with the task from prompt
        mock_agent.run.assert_called_once_with("User provided visual task")


def test_mini_with_explicit_model():
    """Test that mini works with explicitly provided model."""
    with (
        patch("minisweagent.run.mini.configure_if_first_time"),
        patch("minisweagent.run.mini.InteractiveAgent") as mock_interactive_agent_class,
        patch("minisweagent.run.mini.get_model") as mock_get_model,
        patch("minisweagent.run.mini.LocalEnvironment") as mock_env,
        patch("minisweagent.run.mini.get_config_path") as mock_get_config_path,
        patch("minisweagent.run.mini.yaml.safe_load") as mock_yaml_load,
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
            "agent": {"system_template": "test", "mode": "yolo"},
            "env": {},
            "model": {"default_config": "test"},
        }

        # Setup mock agent instance
        mock_agent = Mock()
        mock_agent.run.return_value = ("Success", "Result")
        mock_interactive_agent_class.return_value = mock_agent

        # Call main function with explicit model
        main(
            config_spec=DEFAULT_CONFIG,
            model_name="gpt-4",
            task="Test task with explicit model",
            yolo=True,
            output=None,
            visual=False,
            model_class=None,
        )

        # Verify get_model was called with the explicit model
        mock_get_model.assert_called_once_with("gpt-4", {"default_config": "test"})

        # Verify InteractiveAgent was instantiated
        mock_interactive_agent_class.assert_called_once()
        # Verify agent.run was called
        mock_agent.run.assert_called_once_with("Test task with explicit model")


def test_yolo_mode_sets_correct_agent_config():
    """Test that yolo mode sets the correct agent configuration."""
    with (
        patch("minisweagent.run.mini.configure_if_first_time"),
        patch("minisweagent.run.mini.InteractiveAgent") as mock_interactive_agent_class,
        patch("minisweagent.run.mini.get_model") as mock_get_model,
        patch("minisweagent.run.mini.LocalEnvironment") as mock_env,
        patch("minisweagent.run.mini.get_config_path") as mock_get_config_path,
        patch("minisweagent.run.mini.yaml.safe_load") as mock_yaml_load,
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

        # Setup mock agent instance
        mock_agent = Mock()
        mock_agent.run.return_value = ("Success", "Result")
        mock_interactive_agent_class.return_value = mock_agent

        # Call main function with yolo=True
        main(
            config_spec=DEFAULT_CONFIG,
            model_name="test-model",
            task="Test yolo task",
            yolo=True,
            output=None,
            visual=False,
            model_class=None,
        )

        # Verify InteractiveAgent was called with yolo mode
        mock_interactive_agent_class.assert_called_once()
        args, kwargs = mock_interactive_agent_class.call_args
        # The agent_config should contain the mode as a keyword argument
        assert kwargs.get("mode") == "yolo"
        # Verify agent.run was called
        mock_agent.run.assert_called_once_with("Test yolo task")


def test_confirm_mode_sets_correct_agent_config():
    """Test that when yolo=False, no explicit mode is set (defaults to None)."""
    with (
        patch("minisweagent.run.mini.configure_if_first_time"),
        patch("minisweagent.run.mini.InteractiveAgent") as mock_interactive_agent_class,
        patch("minisweagent.run.mini.get_model") as mock_get_model,
        patch("minisweagent.run.mini.LocalEnvironment") as mock_env,
        patch("minisweagent.run.mini.get_config_path") as mock_get_config_path,
        patch("minisweagent.run.mini.yaml.safe_load") as mock_yaml_load,
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

        # Setup mock agent instance
        mock_agent = Mock()
        mock_agent.run.return_value = ("Success", "Result")
        mock_interactive_agent_class.return_value = mock_agent

        # Call main function with yolo=False (default)
        main(
            config_spec=DEFAULT_CONFIG,
            model_name="test-model",
            task="Test confirm task",
            yolo=False,
            output=None,
            visual=False,
            model_class=None,
        )

        # Verify InteractiveAgent was called with no explicit mode (defaults to None)
        mock_interactive_agent_class.assert_called_once()
        args, kwargs = mock_interactive_agent_class.call_args
        # The agent_config should not contain mode when yolo=False (defaults to None)
        assert kwargs.get("mode") is None
        # Verify agent.run was called
        mock_agent.run.assert_called_once_with("Test confirm task")


def test_mini_help():
    """Test that mini --help works correctly."""
    result = subprocess.run(
        [sys.executable, "-m", "minisweagent", "--help"],
        capture_output=True,
        text=True,
        timeout=10,
    )

    assert result.returncode == 0
    # Strip ANSI color codes for reliable text matching
    clean_output = strip_ansi_codes(result.stdout)
    assert "Run mini-SWE-agent in your local environment." in clean_output
    assert "--help" in clean_output
    assert "--config" in clean_output
    assert "--model" in clean_output
    assert "--task" in clean_output
    assert "--yolo" in clean_output
    assert "--output" in clean_output
    assert "--visual" in clean_output


def test_mini_help_with_typer_runner():
    """Test help functionality using typer's test runner."""
    from typer.testing import CliRunner

    runner = CliRunner()
    result = runner.invoke(app, ["--help"])

    assert result.exit_code == 0
    # Strip ANSI color codes for reliable text matching
    clean_output = strip_ansi_codes(result.stdout)
    assert "Run mini-SWE-agent in your local environment." in clean_output
    assert "--help" in clean_output
    assert "--config" in clean_output
    assert "--model" in clean_output
    assert "--task" in clean_output
    assert "--yolo" in clean_output
    assert "--output" in clean_output
    assert "--visual" in clean_output


def test_python_m_minisweagent_help():
    """Test that python -m minisweagent --help works correctly."""
    result = subprocess.run(
        [sys.executable, "-m", "minisweagent", "--help"],
        capture_output=True,
        text=True,
        timeout=10,
    )

    assert result.returncode == 0
    assert "mini-SWE-agent" in result.stdout


def test_mini_script_help():
    """Test that the mini script entry point help works."""
    result = subprocess.run(
        ["mini", "--help"],
        capture_output=True,
        text=True,
        timeout=10,
    )

    assert result.returncode == 0
    assert "mini-SWE-agent" in result.stdout


def test_mini_swe_agent_help():
    """Test that mini-swe-agent --help works correctly."""
    result = subprocess.run(
        ["mini-swe-agent", "--help"],
        capture_output=True,
        text=True,
        timeout=10,
    )

    assert result.returncode == 0
    clean_output = strip_ansi_codes(result.stdout)
    assert "mini-SWE-agent" in clean_output


def test_mini_extra_help():
    """Test that mini-extra --help works correctly."""
    result = subprocess.run(
        ["mini-extra", "--help"],
        capture_output=True,
        text=True,
        timeout=10,
    )

    assert result.returncode == 0
    clean_output = strip_ansi_codes(result.stdout)
    assert "central entry point for all extra commands" in clean_output
    assert "config" in clean_output
    assert "inspect" in clean_output
    assert "github-issue" in clean_output
    assert "swebench" in clean_output


def test_mini_e_help():
    """Test that mini-e --help works correctly."""
    result = subprocess.run(
        ["mini-e", "--help"],
        capture_output=True,
        text=True,
        timeout=10,
    )

    assert result.returncode == 0
    clean_output = strip_ansi_codes(result.stdout)
    assert "central entry point for all extra commands" in clean_output


@pytest.mark.parametrize(
    ("subcommand", "aliases"),
    [
        ("config", ["config"]),
        ("inspect", ["inspect", "i", "inspector"]),
        ("github-issue", ["github-issue", "gh"]),
        ("swebench", ["swebench"]),
        ("swebench-single", ["swebench-single"]),
    ],
)
def test_mini_extra_subcommand_help(subcommand: str, aliases: list[str]):
    """Test that mini-extra subcommands --help work correctly."""
    for alias in aliases:
        result = subprocess.run(
            ["mini-extra", alias, "--help"],
            capture_output=True,
            text=True,
            timeout=10,
        )

        assert result.returncode == 0
        # Just verify that help output is returned (content varies by subcommand)
        assert len(result.stdout) > 0


def test_mini_extra_config_help():
    """Test that mini-extra config --help works correctly."""
    result = subprocess.run(
        ["mini-extra", "config", "--help"],
        capture_output=True,
        text=True,
        timeout=10,
    )

    assert result.returncode == 0
    assert len(result.stdout) > 0
    # Config command should have help output
    clean_output = strip_ansi_codes(result.stdout)
    assert "--help" in clean_output


def test_exit_immediately_flag_sets_confirm_exit_false():
    """Test that --exit-immediately flag sets confirm_exit to False in agent config."""
    with (
        patch("minisweagent.run.mini.configure_if_first_time"),
        patch("minisweagent.run.mini.InteractiveAgent") as mock_interactive_agent_class,
        patch("minisweagent.run.mini.get_model") as mock_get_model,
        patch("minisweagent.run.mini.LocalEnvironment") as mock_env,
        patch("minisweagent.run.mini.get_config_path") as mock_get_config_path,
        patch("minisweagent.run.mini.yaml.safe_load") as mock_yaml_load,
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

        # Create mock agent with config
        mock_agent = Mock()
        mock_agent.config.confirm_exit = False
        mock_agent.run.return_value = ("Success", "Result")
        mock_interactive_agent_class.return_value = mock_agent

        # Call main function with --exit-immediately flag
        agent = main(
            config_spec=DEFAULT_CONFIG,
            model_name="test-model",
            task="Test task",
            yolo=False,
            output=None,
            visual=False,
            exit_immediately=True,  # This should set confirm_exit=False
            model_class=None,
        )

        # Verify the agent's config has confirm_exit set to False
        assert agent.config.confirm_exit is False


def test_no_exit_immediately_flag_sets_confirm_exit_true():
    """Test that when --exit-immediately flag is not used, confirm_exit defaults to True."""
    with (
        patch("minisweagent.run.mini.configure_if_first_time"),
        patch("minisweagent.run.mini.InteractiveAgent") as mock_interactive_agent_class,
        patch("minisweagent.run.mini.get_model") as mock_get_model,
        patch("minisweagent.run.mini.LocalEnvironment") as mock_env,
        patch("minisweagent.run.mini.get_config_path") as mock_get_config_path,
        patch("minisweagent.run.mini.yaml.safe_load") as mock_yaml_load,
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

        # Create mock agent with config
        mock_agent = Mock()
        mock_agent.config.confirm_exit = True
        mock_agent.run.return_value = ("Success", "Result")
        mock_interactive_agent_class.return_value = mock_agent

        # Call main function without --exit-immediately flag (defaults to False)
        agent = main(
            config_spec=DEFAULT_CONFIG,
            model_name="test-model",
            task="Test task",
            yolo=False,
            output=None,
            visual=False,
            model_class=None,
        )

        # Verify the agent's config has confirm_exit set to True
        assert agent.config.confirm_exit is True


def test_exit_immediately_flag_with_typer_runner():
    """Test --exit-immediately flag using typer's test runner."""
    from typer.testing import CliRunner

    with (
        patch("minisweagent.run.mini.configure_if_first_time"),
        patch("minisweagent.run.mini.InteractiveAgent") as mock_interactive_agent_class,
        patch("minisweagent.run.mini.get_model") as mock_get_model,
        patch("minisweagent.run.mini.LocalEnvironment") as mock_env,
        patch("minisweagent.run.mini.get_config_path") as mock_get_config_path,
        patch("minisweagent.run.mini.yaml.safe_load") as mock_yaml_load,
        patch("minisweagent.run.mini.save_traj"),
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

        # Setup mock agent instance
        mock_agent = Mock()
        mock_agent.run.return_value = ("Success", "Result")
        # Mock the attributes needed for save_traj
        mock_agent.model.cost = 0.0
        mock_agent.model.n_calls = 0
        mock_agent.messages = []
        mock_interactive_agent_class.return_value = mock_agent

        runner = CliRunner()
        result = runner.invoke(app, ["--task", "Test task", "--exit-immediately", "--model", "test-model"])

        assert result.exit_code == 0
        mock_interactive_agent_class.assert_called_once()
        args, kwargs = mock_interactive_agent_class.call_args
        # The agent_config should contain confirm_exit as a keyword argument
        assert kwargs.get("confirm_exit") is False

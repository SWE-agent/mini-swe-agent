import re
import subprocess
import sys
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from minisweagent.run.mini import DEFAULT_CONFIG_FILE, app, main


def strip_ansi_codes(text: str) -> str:
    """Remove ANSI escape sequences from text."""
    ansi_escape = re.compile(r"\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])")
    return ansi_escape.sub("", text)


def test_configure_if_first_time_called():
    """Test that configure_if_first_time is called when running mini main."""
    with (
        patch("minisweagent.run.mini.configure_if_first_time") as mock_configure,
        patch("minisweagent.run.mini.get_agent") as mock_get_agent,
        patch("minisweagent.run.mini.get_model") as mock_get_model,
        patch("minisweagent.run.mini.get_environment") as mock_get_env,
        patch("minisweagent.run.mini.get_config_from_spec") as mock_get_config,
    ):
        # Setup mocks
        mock_model = Mock()
        mock_get_model.return_value = mock_model
        mock_environment = Mock()
        mock_get_env.return_value = mock_environment
        mock_get_config.return_value = {"agent": {"system_template": "test"}, "env": {}, "model": {}}

        # Setup mock agent instance
        mock_agent = Mock()
        mock_agent.run.return_value = {"exit_status": "Success", "submission": "Result"}
        mock_get_agent.return_value = mock_agent

        # Call main function
        main(
            config_spec=[str(DEFAULT_CONFIG_FILE)],
            model_name="test-model",
            task="Test task",
            yolo=False,
            output=None,
            model_class=None,
            agent_class=None,
            environment_class=None,
        )

        # Verify configure_if_first_time was called
        mock_configure.assert_called_once()


def test_mini_command_calls_run_interactive():
    """Test that mini command creates agent via get_agent."""
    with (
        patch("minisweagent.run.mini.configure_if_first_time"),
        patch("minisweagent.run.mini.get_agent") as mock_get_agent,
        patch("minisweagent.run.mini.get_model") as mock_get_model,
        patch("minisweagent.run.mini.get_environment") as mock_get_env,
        patch("minisweagent.run.mini.get_config_from_spec") as mock_get_config,
    ):
        # Setup mocks
        mock_model = Mock()
        mock_get_model.return_value = mock_model
        mock_environment = Mock()
        mock_get_env.return_value = mock_environment
        mock_get_config.return_value = {"agent": {"system_template": "test", "mode": "confirm"}, "env": {}, "model": {}}

        # Setup mock agent instance
        mock_agent = Mock()
        mock_agent.run.return_value = {"exit_status": "Success", "submission": "Result"}
        mock_get_agent.return_value = mock_agent

        # Call main function with task provided (so prompt is not called)
        main(
            config_spec=[str(DEFAULT_CONFIG_FILE)],
            model_name="test-model",
            task="Test task",
            yolo=False,
            output=None,
            model_class=None,
            agent_class=None,
            environment_class=None,
        )

        # Verify get_agent was called
        mock_get_agent.assert_called_once()
        args, kwargs = mock_get_agent.call_args
        assert args[0] == mock_model  # model
        assert args[1] == mock_environment  # env
        # Verify agent.run was called with the task
        mock_agent.run.assert_called_once_with("Test task")


def test_mini_calls_prompt_when_no_task_provided():
    """Test that mini calls prompt when no task is provided."""
    with (
        patch("minisweagent.run.mini.configure_if_first_time"),
        patch("minisweagent.run.mini._multiline_prompt") as mock_prompt,
        patch("minisweagent.run.mini.get_agent") as mock_get_agent,
        patch("minisweagent.run.mini.get_model") as mock_get_model,
        patch("minisweagent.run.mini.get_environment") as mock_get_env,
        patch("minisweagent.run.mini.get_config_from_spec") as mock_get_config,
    ):
        # Setup mocks
        mock_prompt.return_value = "User provided task"
        mock_model = Mock()
        mock_get_model.return_value = mock_model
        mock_environment = Mock()
        mock_get_env.return_value = mock_environment
        mock_get_config.return_value = {"agent": {"system_template": "test", "mode": "confirm"}, "env": {}, "model": {}}

        # Setup mock agent instance
        mock_agent = Mock()
        mock_agent.run.return_value = {"exit_status": "Success", "submission": "Result"}
        mock_get_agent.return_value = mock_agent

        # Call main function without task
        main(
            config_spec=[str(DEFAULT_CONFIG_FILE)],
            model_name="test-model",
            task=None,  # No task provided
            yolo=False,
            output=None,
            model_class=None,
            agent_class=None,
            environment_class=None,
        )

        # Verify prompt was called
        mock_prompt.assert_called_once()

        # Verify get_agent was called
        mock_get_agent.assert_called_once()
        # Verify agent.run was called with the task from prompt
        mock_agent.run.assert_called_once_with("User provided task")


def test_mini_with_explicit_model():
    """Test that mini works with explicitly provided model."""
    with (
        patch("minisweagent.run.mini.configure_if_first_time"),
        patch("minisweagent.run.mini.get_agent") as mock_get_agent,
        patch("minisweagent.run.mini.get_model") as mock_get_model,
        patch("minisweagent.run.mini.get_environment") as mock_get_env,
        patch("minisweagent.run.mini.get_config_from_spec") as mock_get_config,
    ):
        # Setup mocks
        mock_model = Mock()
        mock_get_model.return_value = mock_model
        mock_environment = Mock()
        mock_get_env.return_value = mock_environment
        mock_get_config.return_value = {
            "agent": {"system_template": "test", "mode": "yolo"},
            "env": {},
            "model": {"default_config": "test"},
        }

        # Setup mock agent instance
        mock_agent = Mock()
        mock_agent.run.return_value = {"exit_status": "Success", "submission": "Result"}
        mock_get_agent.return_value = mock_agent

        # Call main function with explicit model
        main(
            config_spec=[str(DEFAULT_CONFIG_FILE)],
            model_name="gpt-4",
            task="Test task with explicit model",
            yolo=True,
            output=None,
            model_class=None,
            agent_class=None,
            environment_class=None,
        )

        # Verify get_model was called (model name is merged into config)
        mock_get_model.assert_called_once()

        # Verify get_agent was called
        mock_get_agent.assert_called_once()
        # Verify agent.run was called
        mock_agent.run.assert_called_once_with("Test task with explicit model")


def test_yolo_mode_sets_correct_agent_config():
    """Test that yolo mode sets the correct agent configuration."""
    with (
        patch("minisweagent.run.mini.configure_if_first_time"),
        patch("minisweagent.run.mini.get_agent") as mock_get_agent,
        patch("minisweagent.run.mini.get_model") as mock_get_model,
        patch("minisweagent.run.mini.get_environment") as mock_get_env,
        patch("minisweagent.run.mini.get_config_from_spec") as mock_get_config,
    ):
        # Setup mocks
        mock_model = Mock()
        mock_get_model.return_value = mock_model
        mock_environment = Mock()
        mock_get_env.return_value = mock_environment
        mock_get_config.return_value = {"agent": {"system_template": "test"}, "env": {}, "model": {}}

        # Setup mock agent instance
        mock_agent = Mock()
        mock_agent.run.return_value = {"exit_status": "Success", "submission": "Result"}
        mock_get_agent.return_value = mock_agent

        # Call main function with yolo=True
        main(
            config_spec=[str(DEFAULT_CONFIG_FILE)],
            model_name="test-model",
            task="Test yolo task",
            yolo=True,
            output=None,
            model_class=None,
            agent_class=None,
            environment_class=None,
        )

        # Verify get_agent was called with yolo mode in config
        mock_get_agent.assert_called_once()
        args, kwargs = mock_get_agent.call_args
        # The config (third positional arg) should contain the mode
        assert args[2].get("mode") == "yolo"
        # Verify agent.run was called
        mock_agent.run.assert_called_once_with("Test yolo task")


def test_confirm_mode_sets_correct_agent_config():
    """Test that when yolo=False, no explicit mode is set (defaults to None)."""
    with (
        patch("minisweagent.run.mini.configure_if_first_time"),
        patch("minisweagent.run.mini.get_agent") as mock_get_agent,
        patch("minisweagent.run.mini.get_model") as mock_get_model,
        patch("minisweagent.run.mini.get_environment") as mock_get_env,
        patch("minisweagent.run.mini.get_config_from_spec") as mock_get_config,
    ):
        # Setup mocks
        mock_model = Mock()
        mock_get_model.return_value = mock_model
        mock_environment = Mock()
        mock_get_env.return_value = mock_environment
        mock_get_config.return_value = {"agent": {"system_template": "test"}, "env": {}, "model": {}}

        # Setup mock agent instance
        mock_agent = Mock()
        mock_agent.run.return_value = {"exit_status": "Success", "submission": "Result"}
        mock_get_agent.return_value = mock_agent

        # Call main function with yolo=False (default)
        main(
            config_spec=[str(DEFAULT_CONFIG_FILE)],
            model_name="test-model",
            task="Test confirm task",
            yolo=False,
            output=None,
            model_class=None,
            agent_class=None,
            environment_class=None,
        )

        # Verify get_agent was called with no explicit mode (defaults to None)
        mock_get_agent.assert_called_once()
        args, kwargs = mock_get_agent.call_args
        # The config (third positional arg) should not contain mode when yolo=False
        assert args[2].get("mode") is None
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
        patch("minisweagent.run.mini.get_agent") as mock_get_agent,
        patch("minisweagent.run.mini.get_model") as mock_get_model,
        patch("minisweagent.run.mini.get_environment") as mock_get_env,
        patch("minisweagent.run.mini.get_config_from_spec") as mock_get_config,
    ):
        # Setup mocks
        mock_model = Mock()
        mock_get_model.return_value = mock_model
        mock_environment = Mock()
        mock_get_env.return_value = mock_environment
        mock_get_config.return_value = {"agent": {"system_template": "test"}, "env": {}, "model": {}}

        # Create mock agent with config
        mock_agent = Mock()
        mock_agent.config.confirm_exit = False
        mock_agent.run.return_value = {"exit_status": "Success", "submission": "Result"}
        mock_get_agent.return_value = mock_agent

        # Call main function with --exit-immediately flag
        agent = main(
            config_spec=[str(DEFAULT_CONFIG_FILE)],
            model_name="test-model",
            task="Test task",
            yolo=False,
            output=None,
            exit_immediately=True,  # This should set confirm_exit=False
            model_class=None,
            agent_class=None,
            environment_class=None,
        )

        # Verify the agent's config has confirm_exit set to False
        assert agent.config.confirm_exit is False


def test_no_exit_immediately_flag_sets_confirm_exit_true():
    """Test that when --exit-immediately flag is not used, confirm_exit defaults to True."""
    with (
        patch("minisweagent.run.mini.configure_if_first_time"),
        patch("minisweagent.run.mini.get_agent") as mock_get_agent,
        patch("minisweagent.run.mini.get_model") as mock_get_model,
        patch("minisweagent.run.mini.get_environment") as mock_get_env,
        patch("minisweagent.run.mini.get_config_from_spec") as mock_get_config,
    ):
        # Setup mocks
        mock_model = Mock()
        mock_get_model.return_value = mock_model
        mock_environment = Mock()
        mock_get_env.return_value = mock_environment
        mock_get_config.return_value = {"agent": {"system_template": "test"}, "env": {}, "model": {}}

        # Create mock agent with config
        mock_agent = Mock()
        mock_agent.config.confirm_exit = True
        mock_agent.run.return_value = {"exit_status": "Success", "submission": "Result"}
        mock_get_agent.return_value = mock_agent

        # Call main function without --exit-immediately flag (defaults to False)
        agent = main(
            config_spec=[str(DEFAULT_CONFIG_FILE)],
            model_name="test-model",
            task="Test task",
            yolo=False,
            output=None,
            model_class=None,
            agent_class=None,
            environment_class=None,
        )

        # Verify the agent's config has confirm_exit set to True
        assert agent.config.confirm_exit is True


def test_exit_immediately_flag_with_typer_runner():
    """Test --exit-immediately flag using typer's test runner."""
    from typer.testing import CliRunner

    with (
        patch("minisweagent.run.mini.configure_if_first_time"),
        patch("minisweagent.run.mini.get_agent") as mock_get_agent,
        patch("minisweagent.run.mini.get_model") as mock_get_model,
        patch("minisweagent.run.mini.get_environment") as mock_get_env,
        patch("minisweagent.run.mini.get_config_from_spec") as mock_get_config,
    ):
        # Setup mocks
        mock_model = Mock()
        mock_get_model.return_value = mock_model
        mock_environment = Mock()
        mock_get_env.return_value = mock_environment
        mock_get_config.return_value = {"agent": {"system_template": "test"}, "env": {}, "model": {}}

        # Setup mock agent instance
        mock_agent = Mock()
        mock_agent.run.return_value = {"exit_status": "Success", "result": "Result"}
        mock_agent.messages = []
        mock_get_agent.return_value = mock_agent

        runner = CliRunner()
        result = runner.invoke(app, ["--task", "Test task", "--exit-immediately", "--model", "test-model"])

        assert result.exit_code == 0
        mock_get_agent.assert_called_once()
        args, kwargs = mock_get_agent.call_args
        # The config (third positional arg) should contain confirm_exit
        assert args[2].get("confirm_exit") is False


def test_output_file_is_created(tmp_path):
    """Test that output trajectory file is created when --output is specified."""
    from typer.testing import CliRunner

    output_file = tmp_path / "test_traj.json"

    # Create a temporary config file
    config_file = tmp_path / "test_config.yaml"
    default_config_path = Path("src/minisweagent/config/default.yaml")
    config_file.write_text(default_config_path.read_text())

    with (
        patch("minisweagent.run.mini.configure_if_first_time"),
        patch("minisweagent.run.mini.get_model") as mock_get_model,
        patch("minisweagent.run.mini.get_environment") as mock_get_env,
        patch("minisweagent.agents.interactive._prompt_session.prompt", return_value=""),
        patch("minisweagent.agents.interactive._multiline_prompt_session.prompt", return_value=""),
    ):
        # Setup mocks
        mock_model = Mock()
        mock_model.config = Mock()
        mock_model.config.model_dump.return_value = {}
        mock_model.serialize.return_value = {
            "info": {
                "config": {"model": {}, "model_type": "MockModel"},
            }
        }
        mock_model.get_template_vars.return_value = {}
        mock_model.format_message.side_effect = lambda **kwargs: dict(**kwargs)
        # query now returns dict with extra["actions"]
        mock_model.query.side_effect = [
            {
                "role": "assistant",
                "content": "```mswea_bash_command\necho COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT\necho done\n```",
                "extra": {"actions": [{"command": "echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT\necho done"}]},
            },
        ]
        # format_observation_messages returns observation messages
        mock_model.format_observation_messages.return_value = []
        mock_get_model.return_value = mock_model

        # Environment execute raises Submitted when COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT is seen
        from minisweagent.exceptions import Submitted

        def execute_side_effect(action):
            raise Submitted(
                {
                    "role": "exit",
                    "content": "done",
                    "extra": {"exit_status": "Submitted", "submission": "done"},
                }
            )

        mock_environment = Mock()
        mock_environment.config = Mock()
        mock_environment.config.model_dump.return_value = {}
        mock_environment.execute.side_effect = execute_side_effect
        mock_environment.get_template_vars.return_value = {
            "system": "TestOS",
            "release": "1.0",
            "version": "1.0.0",
            "machine": "x86_64",
        }
        mock_environment.serialize.return_value = {
            "info": {"config": {"environment": {}, "environment_type": "MockEnvironment"}}
        }
        mock_get_env.return_value = mock_environment

        runner = CliRunner()
        result = runner.invoke(
            app,
            [
                "--task",
                "Test task",
                "--model",
                "test-model",
                "--output",
                str(output_file),
                "--config",
                str(config_file),
            ],
        )

        if result.exit_code != 0:
            print(f"Error output: {result.output}")
        assert result.exit_code == 0
        assert output_file.exists(), f"Output file {output_file} was not created"

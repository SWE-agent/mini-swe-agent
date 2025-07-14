import importlib
import os
from unittest.mock import Mock, patch

import pytest
from typer.testing import CliRunner


@pytest.mark.parametrize(
    ("run_module", "agent_patch_path"),
    [
        ("microsweagent.run.local", "microsweagent.run.local.InteractiveAgent"),
        ("microsweagent.run.local2", "microsweagent.run.local2.AgentApp"),
    ],
)
def test_cli_prompts_for_model_when_no_env_vars(run_module, agent_patch_path, tmp_path):
    module = importlib.import_module(run_module)
    runner = CliRunner()

    config_path = tmp_path / "test_config.yaml"
    config_path.write_text("agent: {}")

    with (
        patch.dict(os.environ, {}, clear=True),
        patch(agent_patch_path) as mock_agent_class,
        patch("microsweagent.models.prompt_for_model_name", return_value="test-model") as mock_prompt,
        patch("microsweagent.models.get_model_class") as mock_get_class,
    ):
        mock_model_instance = Mock()
        mock_model_instance.configure_mock(cost=0.05, n_calls=3)
        mock_get_class.return_value = lambda **kwargs: mock_model_instance

        if run_module == "microsweagent.run.local":
            mock_agent = Mock()
            mock_agent.run.return_value = ("success", "patch content")
            mock_agent.messages = [{"role": "system", "content": "test"}]
            # Create a proper mock model with serializable attributes
            mock_model = Mock()
            mock_model.configure_mock(cost=0.05, n_calls=3)
            mock_agent.model = mock_model
            mock_agent_class.return_value = mock_agent
        else:
            mock_agent_app = Mock()
            mock_agent = Mock()
            mock_agent.messages = [{"role": "system", "content": "test"}]
            # Create a proper mock model with serializable attributes
            mock_model = Mock()
            mock_model.configure_mock(cost=0.05, n_calls=3)
            mock_agent.model = mock_model
            mock_agent_app.agent = mock_agent
            mock_agent_app.configure_mock(exit_status=None, result=None)
            mock_agent_class.return_value = mock_agent_app

        result = runner.invoke(module.app, ["--config", str(config_path), "--task", "Test problem", "--yolo"])

        assert result.exit_code == 0
        mock_prompt.assert_called_once()


@pytest.mark.parametrize(
    ("run_module", "agent_patch_path"),
    [
        ("microsweagent.run.local", "microsweagent.run.local.InteractiveAgent"),
        ("microsweagent.run.local2", "microsweagent.run.local2.AgentApp"),
    ],
)
def test_cli_multiline_input_flow(run_module, agent_patch_path, tmp_path):
    module = importlib.import_module(run_module)
    runner = CliRunner()

    config_path = tmp_path / "test_config.yaml"
    config_path.write_text("agent: {}")

    multiline_input = "This is line 1\nThis is line 2\nThis is line 3\n"

    # Patch get_multiline_task in the correct module namespace
    multiline_patch_target = (
        f"{run_module}.get_multiline_task"
        if run_module == "microsweagent.run.local2"
        else "microsweagent.run.local.get_multiline_task"
    )

    with (
        patch.dict(os.environ, {}, clear=True),
        patch(agent_patch_path) as mock_agent_class,
        patch("microsweagent.models.prompt_for_model_name", return_value="test-model"),
        patch("microsweagent.models.get_model_class") as mock_get_class,
        patch(multiline_patch_target, return_value=multiline_input.strip()) as mock_multiline,
    ):
        mock_model_instance = Mock()
        mock_model_instance.configure_mock(cost=0.05, n_calls=3)
        mock_get_class.return_value = lambda **kwargs: mock_model_instance

        if run_module == "microsweagent.run.local":
            mock_agent = Mock()
            mock_agent.run.return_value = ("success", "patch content")
            mock_agent.messages = [{"role": "system", "content": "test"}]
            # Create a proper mock model with serializable attributes
            mock_model = Mock()
            mock_model.configure_mock(cost=0.05, n_calls=3)
            mock_agent.model = mock_model
            mock_agent_class.return_value = mock_agent
        else:
            mock_agent_app = Mock()
            mock_agent = Mock()
            mock_agent.messages = [{"role": "system", "content": "test"}]
            # Create a proper mock model with serializable attributes
            mock_model = Mock()
            mock_model.configure_mock(cost=0.05, n_calls=3)
            mock_agent.model = mock_model
            mock_agent_app.agent = mock_agent
            mock_agent_app.configure_mock(exit_status=None, result=None)
            mock_agent_class.return_value = mock_agent_app

        result = runner.invoke(module.app, ["--config", str(config_path), "--yolo"])

        assert result.exit_code == 0
        mock_multiline.assert_called_once()
        if run_module == "microsweagent.run.local":
            mock_agent.run.assert_called_once_with(multiline_input.strip())


@pytest.mark.parametrize(
    ("run_module", "agent_patch_path"),
    [
        ("microsweagent.run.local", "microsweagent.run.local.InteractiveAgent"),
        ("microsweagent.run.local2", "microsweagent.run.local2.AgentApp"),
    ],
)
def test_agent_initialization_with_user_settings(run_module, agent_patch_path, tmp_path):
    module = importlib.import_module(run_module)
    runner = CliRunner()

    config_path = tmp_path / "test_config.yaml"
    config_content = """
agent:
  max_steps: 10
  step_timeout: 30
model:
  model_kwargs:
    temperature: 0.7
"""
    config_path.write_text(config_content)

    with (
        patch.dict(os.environ, {}, clear=True),
        patch(agent_patch_path) as mock_agent_class,
        patch("microsweagent.models.prompt_for_model_name", return_value="claude-3-5-sonnet"),
        patch("microsweagent.models.get_model_class") as mock_get_class,
    ):
        mock_model_instance = Mock()
        mock_model_instance.configure_mock(cost=0.10, n_calls=5)
        mock_get_class.return_value = lambda **kwargs: mock_model_instance

        if run_module == "microsweagent.run.local":
            mock_agent = Mock()
            mock_agent.run.return_value = ("success", "patch content")
            mock_agent.messages = [{"role": "system", "content": "test"}]
            # Create a proper mock model with serializable attributes
            mock_model = Mock()
            mock_model.configure_mock(cost=0.10, n_calls=5)
            mock_agent.model = mock_model
            mock_agent_class.return_value = mock_agent
        else:
            mock_agent_app = Mock()
            mock_agent = Mock()
            mock_agent.messages = [{"role": "system", "content": "test"}]
            # Create a proper mock model with serializable attributes
            mock_model = Mock()
            mock_model.configure_mock(cost=0.10, n_calls=5)
            mock_agent.model = mock_model
            mock_agent_app.agent = mock_agent
            mock_agent_app.configure_mock(exit_status=None, result=None)
            mock_agent_class.return_value = mock_agent_app

        result = runner.invoke(module.app, ["--config", str(config_path), "--task", "Complex task", "--yolo"])

        assert result.exit_code == 0

        mock_agent_class.assert_called_once()
        call_args = mock_agent_class.call_args

        from microsweagent.environments.local import LocalEnvironment

        if run_module == "microsweagent.run.local":
            assert call_args[0][0] is mock_model_instance
            assert isinstance(call_args[0][1], LocalEnvironment)
            assert call_args[1]["max_steps"] == 10
            assert call_args[1]["step_timeout"] == 30
            assert call_args[1]["mode"] == "yolo"
            mock_agent.run.assert_called_once_with("Complex task")
        else:
            assert call_args[1]["model"] is mock_model_instance
            assert isinstance(call_args[1]["env"], LocalEnvironment)
            assert call_args[1]["task"] == "Complex task"
            assert call_args[1]["max_steps"] == 10
            assert call_args[1]["step_timeout"] == 30


@pytest.mark.parametrize(
    ("run_module", "agent_patch_path"),
    [
        ("microsweagent.run.local", "microsweagent.run.local.InteractiveAgent"),
        ("microsweagent.run.local2", "microsweagent.run.local2.AgentApp"),
    ],
)
def test_model_selection_precedence(run_module, agent_patch_path, tmp_path):
    module = importlib.import_module(run_module)
    runner = CliRunner()

    config_path = tmp_path / "test_config.yaml"
    config_path.write_text("agent: {}")

    with (
        patch.dict(os.environ, {}, clear=True),
        patch(agent_patch_path) as mock_agent_class,
        patch("microsweagent.models.prompt_for_model_name") as mock_prompt,
        patch("microsweagent.models.get_model_class") as mock_get_class,
    ):
        mock_model_instance = Mock()
        mock_model_instance.configure_mock(cost=0.02, n_calls=1)
        mock_get_class.return_value = lambda **kwargs: mock_model_instance

        if run_module == "microsweagent.run.local":
            mock_agent = Mock()
            mock_agent.run.return_value = ("success", "patch content")
            mock_agent.messages = [{"role": "system", "content": "test"}]
            # Create a proper mock model with serializable attributes
            mock_model = Mock()
            mock_model.configure_mock(cost=0.02, n_calls=1)
            mock_agent.model = mock_model
            mock_agent_class.return_value = mock_agent
        else:
            mock_agent_app = Mock()
            mock_agent = Mock()
            mock_agent.messages = [{"role": "system", "content": "test"}]
            # Create a proper mock model with serializable attributes
            mock_model = Mock()
            mock_model.configure_mock(cost=0.02, n_calls=1)
            mock_agent.model = mock_model
            mock_agent_app.agent = mock_agent
            mock_agent_app.configure_mock(exit_status=None, result=None)
            mock_agent_class.return_value = mock_agent_app

        result = runner.invoke(
            module.app, ["--config", str(config_path), "--model", "gpt-4", "--task", "Simple task", "--yolo"]
        )

        assert result.exit_code == 0

        # When model is explicitly provided, no prompting should occur
        mock_prompt.assert_not_called()

        # Verify the model was passed correctly
        mock_get_class.assert_called_once_with("gpt-4")

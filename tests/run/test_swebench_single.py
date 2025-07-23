from unittest.mock import patch

import pytest
import yaml

from minisweagent import package_dir
from minisweagent.models.test_models import DeterministicModel
from minisweagent.run.extra.swebench_single import main


@pytest.mark.slow
def test_swebench_single_end_to_end(github_test_data):
    """Test the swebench_single script using the _test subset with deterministic model"""

    model_responses = github_test_data["model_responses"]

    with patch("minisweagent.run.extra.swebench_single.get_model") as mock_get_model:
        with patch("minisweagent.agents.interactive.prompt_session.prompt", return_value=""):  # No new task
            mock_get_model.return_value = DeterministicModel(outputs=model_responses, cost_per_call=0.1)

            # Test with explicit instance ID
            main(
                subset="_test",
                split="test",
                instance_spec="swe-agent__test-repo-1",
                model_name="deterministic",
                config_path=package_dir / "config" / "extra" / "swebench.yaml",
            )

        # Verify model was called with correct parameters
        mock_get_model.assert_called_once()


@pytest.mark.slow
def test_swebench_single_with_custom_config(github_test_data, tmp_path):
    """Test swebench_single with custom configuration file"""

    model_responses = github_test_data["model_responses"]

    # Create a custom config file
    custom_config = {"model": {"temperature": 0.5}, "agent": {"step_limit": 10}}
    config_file = tmp_path / "custom_config.yaml"
    config_file.write_text(yaml.dump(custom_config))

    with patch("minisweagent.run.extra.swebench_single.get_model") as mock_get_model:
        with patch("minisweagent.agents.interactive.prompt_session.prompt", return_value=""):  # No new task
            mock_get_model.return_value = DeterministicModel(outputs=model_responses, cost_per_call=0.1)

            main(
                subset="_test",
                split="test",
                instance_spec="swe-agent__test-repo-1",
                model_name="deterministic",
                config_path=config_file,
            )

        # Verify model was called with config
        mock_get_model.assert_called_once()


@pytest.mark.slow
def test_swebench_single_default_parameters(github_test_data):
    """Test swebench_single with default parameter values"""

    model_responses = github_test_data["model_responses"]

    with patch("minisweagent.run.extra.swebench_single.get_model") as mock_get_model:
        with patch("minisweagent.agents.interactive.prompt_session.prompt", return_value=""):  # No new task
            mock_get_model.return_value = DeterministicModel(outputs=model_responses, cost_per_call=0.1)

            # Test with all parameters explicitly provided
            main(
                subset="_test",
                split="test",
                instance_spec="swe-agent__test-repo-1",
                model_name=None,
                config_path=package_dir / "config" / "extra" / "swebench.yaml",
            )

            # Verify model was called
            mock_get_model.assert_called_once()

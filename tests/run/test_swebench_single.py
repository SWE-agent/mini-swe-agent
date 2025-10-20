from unittest.mock import patch

import pytest

from minisweagent import package_dir
from minisweagent.models.test_models import DeterministicModel
from minisweagent.run.extra.swebench_single import main


@pytest.mark.slow
def test_swebench_single_end_to_end(github_test_data, tmp_path):
    """Test the swebench_single script using the _test subset with deterministic model.
    This mostly tests that no exception occurs.
    """

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
                environment_class="docker",
                exit_immediately=False,
                output=tmp_path / "test_output.json",
            )

        # Verify model was called with correct parameters
        mock_get_model.assert_called_once()


@pytest.mark.slow
def test_swebench_single_end_to_end_exit_immediately(github_test_data, tmp_path):
    """Test the swebench_single script using the _test subset with deterministic model.
    This mostly tests that no exception occurs.
    This test uses the --exit-immediately flag to exit immediately when the agent wants to finish instead of prompting.
    """

    model_responses = github_test_data["model_responses"]

    with patch("minisweagent.run.extra.swebench_single.get_model") as mock_get_model:
        mock_get_model.return_value = DeterministicModel(outputs=model_responses, cost_per_call=0.1)

        # Test with explicit instance ID
        main(
            subset="_test",
            split="test",
            instance_spec="swe-agent__test-repo-1",
            model_name="deterministic",
            config_path=package_dir / "config" / "extra" / "swebench.yaml",
            environment_class="docker",
            exit_immediately=True,
            output=tmp_path / "test_output.json",
        )

        # Verify model was called with correct parameters
        mock_get_model.assert_called_once()

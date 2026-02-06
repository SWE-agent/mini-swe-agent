import re
from unittest.mock import patch

import pytest

from minisweagent import package_dir
from minisweagent.models.test_models import DeterministicModel, make_output
from minisweagent.run.benchmarks.swebench_single import main


def _make_model_from_fixture(text_outputs: list[str], cost_per_call: float = 1.0, **kwargs) -> DeterministicModel:
    """Create a DeterministicModel from trajectory fixture data (raw text outputs)."""

    def parse_command(text: str) -> list[dict]:
        match = re.search(r"```mswea_bash_command\s*\n(.*?)\n```", text, re.DOTALL)
        return [{"command": match.group(1)}] if match else []

    return DeterministicModel(
        outputs=[make_output(text, parse_command(text), cost=cost_per_call) for text in text_outputs],
        cost_per_call=cost_per_call,
        **kwargs,
    )


@pytest.mark.slow
def test_swebench_single_end_to_end(github_test_data, tmp_path):
    """Test the swebench_single script using the _test subset with deterministic model.
    This mostly tests that no exception occurs.
    """

    model_responses = github_test_data["model_responses"]

    with (
        patch("minisweagent.run.benchmarks.swebench_single.get_model") as mock_get_model,
        patch("minisweagent.agents.interactive._prompt_session.prompt", side_effect=lambda *a, **kw: ""),
        patch("minisweagent.agents.interactive._multiline_prompt_session.prompt", side_effect=lambda *a, **kw: ""),
        patch("builtins.input", return_value=""),  # For LimitsExceeded handling
    ):
        mock_get_model.return_value = _make_model_from_fixture(model_responses, cost_per_call=0.1)

        # Test with explicit instance ID
        output_path = tmp_path / "test_output.json"
        main(
            subset="_test",
            split="test",
            instance_spec="swe-agent__test-repo-1",
            model_name="deterministic",
            config_spec=[str(package_dir / "config" / "benchmarks" / "swebench.yaml")],
            environment_class="docker",
            exit_immediately=False,
            output=output_path,
            model_class=None,
            agent_class=None,
            yolo=False,
            cost_limit=None,
        )

        # Verify model was called with correct parameters
        mock_get_model.assert_called_once()
        assert output_path.exists()


@pytest.mark.slow
def test_swebench_single_end_to_end_exit_immediately(github_test_data, tmp_path):
    """Test the swebench_single script using the _test subset with deterministic model.
    This mostly tests that no exception occurs.
    This test uses the --exit-immediately flag to exit immediately when the agent wants to finish instead of prompting.
    """

    model_responses = github_test_data["model_responses"]

    with (
        patch("minisweagent.run.benchmarks.swebench_single.get_model") as mock_get_model,
        patch("minisweagent.agents.interactive._prompt_session.prompt", side_effect=lambda *a, **kw: ""),
        patch("minisweagent.agents.interactive._multiline_prompt_session.prompt", side_effect=lambda *a, **kw: ""),
        patch("builtins.input", return_value=""),  # For LimitsExceeded handling
    ):
        mock_get_model.return_value = _make_model_from_fixture(model_responses, cost_per_call=0.1)

        # Test with explicit instance ID
        output_path = tmp_path / "test_output.json"
        main(
            subset="_test",
            split="test",
            instance_spec="swe-agent__test-repo-1",
            model_name="deterministic",
            config_spec=[str(package_dir / "config" / "benchmarks" / "swebench.yaml")],
            environment_class="docker",
            exit_immediately=True,
            output=output_path,
            model_class=None,
            agent_class=None,
            yolo=False,
            cost_limit=None,
        )

        # Verify model was called with correct parameters
        mock_get_model.assert_called_once()
        assert output_path.exists()

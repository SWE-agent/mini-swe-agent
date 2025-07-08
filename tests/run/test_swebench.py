import json
from unittest.mock import patch

import pytest

from microswea import package_dir
from microswea.models.test_models import DeterministicModel
from microswea.run.extra.swebench import main


@pytest.mark.slow
@pytest.mark.parametrize("n_workers", [1, 2])
def test_swebench_end_to_end(github_test_data, tmp_path, n_workers):
    """Test the complete SWEBench flow using the _test subset with deterministic model"""

    model_responses = github_test_data["model_responses"]
    output_file = tmp_path / "results.json"

    with patch("microswea.run.extra.swebench.get_model") as mock_get_model:
        mock_get_model.return_value = DeterministicModel(outputs=model_responses)

        main(subset="_test", split="test", slice_spec="0:1", output=str(output_file), n_workers=n_workers)

    # Extract the last observation from github_issue.traj.json
    traj_file_path = package_dir.parent.parent / "tests" / "test_data" / "github_issue.traj.json"
    with open(traj_file_path) as f:
        trajectory = json.load(f)

    # Get the last message content as the model patch
    last_message = trajectory[-1]["content"]

    # Expected format
    expected_result = {
        "swe-agent__test-repo-1": {
            "model_name_or_path": "deterministic",
            "instance_id": "swe-agent__test-repo-1",
            "model_patch": last_message,
        }
    }

    # Load and check actual results
    with open(output_file) as f:
        actual_result = json.load(f)

    assert actual_result == expected_result

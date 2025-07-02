import json
from unittest.mock import patch

from nanoswea import package_dir
from nanoswea.extra.model.test_models import DeterministicModel, DeterministicModelConfig
from nanoswea.extra.run.run_swebench import main


def test_swebench_end_to_end(test_data, tmp_path):
    """Test the complete SWEBench flow using the _test subset with deterministic model"""

    model_responses = test_data["model_responses"]
    output_file = tmp_path / "results.json"

    with patch("nanoswea.extra.run.run_swebench.LitellmModel") as mock_model_class:
        mock_model_class.return_value = DeterministicModel(DeterministicModelConfig(outputs=model_responses))

        main(subset="_test", split="test", slice_spec="0:1", output=str(output_file))

    # Extract the last observation from github_issue.traj.json
    traj_file_path = package_dir.parent / "tests" / "test_data" / "github_issue.traj.json"
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

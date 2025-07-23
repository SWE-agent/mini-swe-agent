#!/usr/bin/env python3

import json
import sys
from pathlib import Path
from unittest.mock import patch

from minisweagent.models.test_models import DeterministicModel
from minisweagent.run.github_issue import DEFAULT_CONFIG, main


def update_trajectory():
    traj_path = Path(__file__).parent / "github_issue.traj.json"
    trajectory = json.loads(traj_path.read_text())

    model_responses = [msg["content"] for msg in trajectory[2:] if msg["role"] == "assistant"]

    with patch("minisweagent.run.github_issue.get_model") as mock_get_model:
        mock_get_model.return_value = DeterministicModel(outputs=model_responses)
        github_url = "https://github.com/SWE-agent/test-repo/issues/1"
        agent = main(issue_url=github_url, model="tardis", config=DEFAULT_CONFIG)

    traj_path.write_text(json.dumps(agent.messages, indent=2))


if __name__ == "__main__":
    update_trajectory()
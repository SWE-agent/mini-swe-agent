#!/usr/bin/env python3
"""
Update test trajectory files by replaying them with DeterministicModel.

This script takes a trajectory JSON file, extracts the model responses,
and replays the agent run with those exact responses to generate an
updated trajectory file that matches the current agent behavior.

Usage:
    python tests/test_data/update_outputs.py <path_to_trajectory.json>
"""

import json
import sys
from pathlib import Path

from microswea.agents.default import DefaultAgent
from microswea.environments.docker import DockerEnvironment
from microswea.models.test_models import DeterministicModel


def main():
    traj_path = Path(sys.argv[1])

    trajectory = json.loads(traj_path.read_text())

    problem_statement = trajectory[1]["content"]
    model_responses = [msg["content"] for msg in trajectory[2:] if msg["role"] == "assistant"]

    model = DeterministicModel(outputs=model_responses)
    agent = DefaultAgent(model, DockerEnvironment(image="python:3.11"), problem_statement)

    agent.run()

    traj_path.write_text(json.dumps(agent.messages, indent=2))


if __name__ == "__main__":
    main()


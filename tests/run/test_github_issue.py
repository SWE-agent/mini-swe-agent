import re
from pathlib import Path
from unittest.mock import Mock, patch

import pytest
import yaml

from minisweagent.models.test_models import DeterministicModel
from minisweagent.run.extra.github_issue import DEFAULT_CONFIG, main


def normalize_outputs(s: str) -> str:
    """Strip leading/trailing whitespace and normalize internal whitespace"""
    # Remove everything between <args> and </args>, because this contains docker container ids
    s = re.sub(r"<args>(.*?)</args>", "", s, flags=re.DOTALL)
    # Replace all lines that have root in them because they tend to appear with times
    s = "\n".join(l for l in s.split("\n") if "root root" not in l)
    return "\n".join(line.rstrip() for line in s.strip().split("\n"))


def assert_observations_match(expected_observations: list[str], messages: list[dict]) -> None:
    """Compare expected observations with actual observations from agent messages

    Args:
        expected_observations: List of expected observation strings
        messages: Agent conversation messages (list of message dicts with 'role' and 'content')
    """
    # Extract actual observations from agent messages
    # User/exit messages (observations) are at indices 3, 5, 7, etc.
    actual_observations = []
    for i in range(len(expected_observations)):
        user_message_index = 3 + (i * 2)
        assert messages[user_message_index]["role"] in ("user", "exit")
        actual_observations.append(messages[user_message_index]["content"])

    assert len(actual_observations) == len(expected_observations), (
        f"Expected {len(expected_observations)} observations, got {len(actual_observations)}"
    )

    for i, (expected_observation, actual_observation) in enumerate(zip(expected_observations, actual_observations)):
        normalized_actual = normalize_outputs(actual_observation)
        normalized_expected = normalize_outputs(expected_observation)

        assert normalized_actual == normalized_expected, (
            f"Step {i + 1} observation mismatch:\nExpected: {repr(normalized_expected)}\nActual: {repr(normalized_actual)}"
        )


def test_configure_if_first_time_called():
    """Test that configure_if_first_time is called when running github_issue main."""
    with (
        patch("minisweagent.run.extra.github_issue.configure_if_first_time") as mock_configure,
        patch("minisweagent.run.extra.github_issue.fetch_github_issue") as mock_fetch,
        patch("minisweagent.run.extra.github_issue.InteractiveAgent") as mock_agent,
        patch("minisweagent.run.extra.github_issue.get_model"),
        patch("minisweagent.run.extra.github_issue.DockerEnvironment"),
        patch("minisweagent.run.extra.github_issue.yaml.safe_load") as mock_yaml_load,
        patch("minisweagent.run.extra.github_issue.get_config_path") as mock_get_config_path,
    ):
        mock_fetch.return_value = "Test issue"
        mock_yaml_load.return_value = {"agent": {}, "environment": {}, "model": {}}
        mock_get_config_path.return_value.read_text.return_value = "test config"
        mock_agent_instance = mock_agent.return_value
        mock_agent_instance.run.return_value = {"exit_status": "Submitted", "submission": "success"}
        mock_agent_instance.env.execute.return_value = None

        main(issue_url="https://github.com/test/repo/issues/1", config=DEFAULT_CONFIG, model="test-model", yolo=True)

        mock_configure.assert_called_once()


def test_output_file_is_created(tmp_path):
    """Test that output trajectory file is created when output is specified."""
    output_file = tmp_path / "test_github_traj.json"

    # Create a temporary config file with output_path set
    config_file = tmp_path / "test_config.yaml"
    default_config_path = Path("src/minisweagent/config/github_issue.yaml")
    config = yaml.safe_load(default_config_path.read_text())
    config["agent"]["output_path"] = str(output_file)
    config_file.write_text(yaml.dump(config))

    with (
        patch("minisweagent.run.extra.github_issue.configure_if_first_time"),
        patch("minisweagent.run.extra.github_issue.fetch_github_issue") as mock_fetch,
        patch("minisweagent.run.extra.github_issue.get_model") as mock_get_model,
        patch("minisweagent.run.extra.github_issue.DockerEnvironment") as mock_env_class,
        patch("minisweagent.agents.interactive.prompt_session.prompt", return_value=""),
    ):
        mock_fetch.return_value = "Test issue"

        # Setup mock model and environment with required attributes
        mock_model = Mock()
        mock_model.cost = 0.0
        mock_model.n_calls = 0
        mock_model.config = Mock()
        mock_model.config.model_dump.return_value = {}
        mock_model.serialize.return_value = {
            "info": {
                "model_stats": {"instance_cost": 0.0, "api_calls": 0},
                "config": {"model": {}, "model_type": "MockModel"},
            }
        }
        mock_model.get_template_vars.return_value = {}
        # query now returns list[dict] with action key
        mock_model.query.side_effect = [
            [
                {
                    "role": "assistant",
                    "content": "```bash\necho COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT\ndone\n```",
                    "action": "echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT\ndone",
                }
            ],
        ]
        mock_get_model.return_value = mock_model

        mock_env = Mock()
        mock_env.config = Mock()
        mock_env.config.model_dump.return_value = {}
        mock_env.execute.return_value = {"output": "COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT\ndone", "returncode": 0}
        # execute_messages returns list of observation messages and raises Submitted for completion
        from minisweagent.exceptions import Submitted

        mock_env.execute_messages.side_effect = Submitted(
            {
                "role": "exit",
                "content": "done",
                "extra": {"exit_status": "Submitted", "submission": "done"},
            }
        )
        mock_env.get_template_vars.return_value = {
            "system": "TestOS",
            "release": "1.0",
            "version": "1.0.0",
            "machine": "x86_64",
        }
        mock_env.serialize.return_value = {
            "info": {"config": {"environment": {}, "environment_type": "MockEnvironment"}}
        }
        mock_env_class.return_value = mock_env

        main(
            issue_url="https://github.com/test/repo/issues/1",
            config=config_file,
            model="test-model",
            yolo=True,
        )

        assert output_file.exists(), f"Output file {output_file} was not created"


@pytest.mark.slow
def test_github_issue_end_to_end(github_test_data):
    """Test the complete flow from CLI to final result using real environment but deterministic model"""

    model_responses = github_test_data["model_responses"]
    expected_observations = github_test_data["expected_observations"]

    with (
        patch("minisweagent.run.extra.github_issue.configure_if_first_time"),
        patch("minisweagent.run.extra.github_issue.get_model") as mock_get_model,
        patch("minisweagent.agents.interactive.prompt_session.prompt", return_value=""),  # No new task
    ):
        mock_get_model.return_value = DeterministicModel(outputs=model_responses)
        github_url = "https://github.com/SWE-agent/test-repo/issues/1"
        agent = main(issue_url=github_url, model="tardis", config=DEFAULT_CONFIG, yolo=True)  # type: ignore

    assert agent is not None
    messages = agent.messages

    # Verify we have the right number of messages
    # Should be: system + user (initial) + (assistant + user) * number_of_steps
    expected_total_messages = 2 + (len(model_responses) * 2)
    assert len(messages) == expected_total_messages, f"Expected {expected_total_messages} messages, got {len(messages)}"

    assert_observations_match(expected_observations, messages)

    assert agent.model.n_calls == len(model_responses), (
        f"Expected {len(model_responses)} steps, got {agent.model.n_calls}"
    )

import re
from unittest.mock import patch

import pytest

from minisweagent.models.test_models import DeterministicModel
from minisweagent.run.github_issue import DEFAULT_CONFIG, main


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
    # User messages (observations) are at indices 3, 5, 7, etc.
    actual_observations = []
    for i in range(len(expected_observations)):
        user_message_index = 3 + (i * 2)
        assert messages[user_message_index]["role"] == "user"
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
        patch("minisweagent.run.github_issue.configure_if_first_time") as mock_configure,
        patch("minisweagent.run.github_issue.fetch_github_issue") as mock_fetch,
        patch("minisweagent.run.github_issue.InteractiveAgent") as mock_agent,
        patch("minisweagent.run.github_issue.get_model"),
        patch("minisweagent.run.github_issue.DockerEnvironment"),
        patch("minisweagent.run.github_issue.yaml.safe_load") as mock_yaml_load,
        patch("minisweagent.run.github_issue.get_config_path") as mock_get_config_path,
        patch("minisweagent.run.github_issue.save_traj"),
    ):
        mock_fetch.return_value = "Test issue"
        mock_yaml_load.return_value = {"agent": {}, "environment": {}, "model": {}}
        mock_get_config_path.return_value.read_text.return_value = "test config"
        mock_agent_instance = mock_agent.return_value
        mock_agent_instance.run.return_value = (0, "success")
        mock_agent_instance.env.execute.return_value = None

        main(issue_url="https://github.com/test/repo/issues/1", config=DEFAULT_CONFIG, model="test-model", yolo=True)

        mock_configure.assert_called_once()


@pytest.mark.slow
def test_github_issue_end_to_end(github_test_data):
    """Test the complete flow from CLI to final result using real environment but deterministic model"""

    model_responses = github_test_data["model_responses"]
    expected_observations = github_test_data["expected_observations"]

    with (
        patch("minisweagent.run.github_issue.configure_if_first_time"),
        patch("minisweagent.run.github_issue.get_model") as mock_get_model,
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

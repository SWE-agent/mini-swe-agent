from unittest.mock import patch

from minisweagent.models.test_models import DeterministicModel
from minisweagent.run.mini import DEFAULT_CONFIG, main
from tests.run.test_github_issue import assert_observations_match


def test_local_end_to_end(local_test_data):
    """Test the complete flow from CLI to final result using real environment but deterministic model"""

    model_responses = local_test_data["model_responses"]
    expected_observations = local_test_data["expected_observations"]

    with (
        patch("minisweagent.run.mini.configure_if_first_time"),
        patch("minisweagent.models.litellm_model.LitellmModel") as mock_model_class,
        patch("minisweagent.agents.interactive.prompt_session.prompt", return_value=""),  # No new task
    ):
        mock_model_class.return_value = DeterministicModel(outputs=model_responses)
        agent = main(
            model_name="tardis",
            config_spec=DEFAULT_CONFIG,
            yolo=True,
            task="Blah blah blah",
            output=None,
            visual=False,
            cost_limit=10,
            model_class=None,
        )  # type: ignore

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

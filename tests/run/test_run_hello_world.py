from unittest.mock import patch

from minisweagent.models.test_models import DeterministicModel
from minisweagent.run.hello_world import main
from tests.run.test_github_issue import assert_observations_match


def test_run_hello_world_end_to_end(local_test_data):
    """Test the complete flow from CLI to final result using real environment but deterministic model"""

    model_responses = local_test_data["model_responses"]
    expected_observations = local_test_data["expected_observations"]

    with (
        patch("minisweagent.run.hello_world.LitellmModel") as mock_model_class,
        patch("os.environ", {"MSWEA_MODEL_NAME": "tardis"}),
    ):
        mock_model_class.return_value = DeterministicModel(outputs=model_responses)
        agent = main(task="Blah blah blah")

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

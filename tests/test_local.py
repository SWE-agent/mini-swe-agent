import re
from unittest.mock import patch

from nanoswea.extra.model.test_models import DeterministicModel, DeterministicModelConfig
from nanoswea.run_local import DEFAULT_CONFIG, main
from tests.test_github_issue import assert_observations_match


def test_local_end_to_end(local_test_data):
    """Test the complete flow from CLI to final result using real environment but deterministic model"""

    model_responses = local_test_data["model_responses"]
    expected_observations = local_test_data["expected_observations"]

    with patch("nanoswea.run_local.LitellmModel") as mock_model_class:
        mock_model_class.return_value = DeterministicModel(DeterministicModelConfig(outputs=model_responses))
        agent = main(model="tardis", config=DEFAULT_CONFIG, yolo=True, problem="Blah blah blah")  # type: ignore

    assert agent is not None
    history = agent.history

    # Verify we have the right number of messages
    # Should be: system + user (initial) + (assistant + user) * number_of_steps
    expected_total_messages = 2 + (len(model_responses) * 2)
    assert len(history) == expected_total_messages, f"Expected {expected_total_messages} messages, got {len(history)}"

    assert_observations_match(expected_observations, history)

    assert agent.model.n_calls == len(model_responses), (
        f"Expected {len(model_responses)} steps, got {agent.model.n_calls}"
    )

import re
from unittest.mock import patch

from minisweagent.models.test_models import DeterministicModel, make_output
from minisweagent.run.hello_world import main
from tests.conftest import assert_observations_match


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


def test_run_hello_world_end_to_end(local_test_data):
    """Test the complete flow from CLI to final result using real environment but deterministic model"""

    model_responses = local_test_data["model_responses"]
    expected_observations = local_test_data["expected_observations"]

    with (
        patch("minisweagent.run.hello_world.LitellmModel") as mock_model_class,
        patch("os.environ", {"MSWEA_MODEL_NAME": "tardis"}),
    ):
        mock_model_class.return_value = _make_model_from_fixture(model_responses)
        agent = main(task="Blah blah blah")

    assert agent is not None
    messages = agent.messages

    # Verify we have the right number of messages
    # Should be: system + user (initial) + (assistant + user) * number_of_steps
    expected_total_messages = 2 + (len(model_responses) * 2)
    assert len(messages) == expected_total_messages, f"Expected {expected_total_messages} messages, got {len(messages)}"

    assert_observations_match(expected_observations, messages)

    assert agent.n_calls == len(model_responses), f"Expected {len(model_responses)} steps, got {agent.n_calls}"

import re
from unittest.mock import patch

from minisweagent.models.test_models import DeterministicModel, make_output
from minisweagent.run.mini import DEFAULT_CONFIG_FILE, main
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


def test_local_end_to_end(local_test_data):
    """Test the complete flow from CLI to final result using real environment but deterministic model"""

    model_responses = local_test_data["model_responses"]
    expected_observations = local_test_data["expected_observations"]

    with (
        patch("minisweagent.run.mini.configure_if_first_time"),
        patch("minisweagent.models.litellm_model.LitellmModel") as mock_model_class,
        patch("minisweagent.agents.interactive._prompt_session.prompt", side_effect=lambda *a, **kw: ""),
        patch("minisweagent.agents.interactive._multiline_prompt_session.prompt", side_effect=lambda *a, **kw: ""),
        patch("builtins.input", return_value=""),  # For LimitsExceeded handling
    ):
        mock_model_class.return_value = _make_model_from_fixture(model_responses)
        agent = main(
            model_name="tardis",
            config_spec=[str(DEFAULT_CONFIG_FILE)],
            yolo=True,
            task="Blah blah blah",
            output=None,
            cost_limit=10,
            model_class=None,
            agent_class=None,
            environment_class=None,
        )  # type: ignore

    assert agent is not None
    messages = agent.messages

    # Verify we have the right number of messages
    # Should be: system + user (initial) + (assistant + user) * number_of_steps
    expected_total_messages = 2 + (len(model_responses) * 2)
    assert len(messages) == expected_total_messages, f"Expected {expected_total_messages} messages, got {len(messages)}"

    assert_observations_match(expected_observations, messages)

    assert agent.n_calls == len(model_responses), f"Expected {len(model_responses)} steps, got {agent.n_calls}"

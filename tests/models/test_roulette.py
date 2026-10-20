import json

import pytest

from minisweagent.agents.default import DefaultAgent
from minisweagent.environments.local import LocalEnvironment
from minisweagent.models.extra.roulette import InterleavingModel, RouletteModel
from minisweagent.models.test_models import make_output
from minisweagent.models.utils.openai_multimodal import DEFAULT_MULTIMODAL_REGEX


@pytest.mark.parametrize(("model_class",), [(RouletteModel,), (InterleavingModel,)])  # noqa: PT006
def test_composite_model_runs_with_default_agent(model_class, tmp_path, reset_global_stats):
    model = model_class(
        model_kwargs=[
            {
                "model_class": "deterministic",
                "model_name": "child",
                "outputs": [make_output("run command", [{"command": "echo hello"}])],
                "observation_template": "child: {{output.output}}",
                "multimodal_regex": DEFAULT_MULTIMODAL_REGEX,
            }
        ]
    )
    agent = DefaultAgent(
        model,
        LocalEnvironment(cwd=str(tmp_path)),
        system_template="system",
        instance_template="{{task}}",
        step_limit=1,
        output_path=tmp_path / "run.traj.json",
    )

    assert agent.run("task")["exit_status"] == "LimitsExceeded"
    assert agent.messages[0]["content"] == [{"type": "text", "text": "system"}]
    assert agent.messages[3]["content"] == [{"type": "text", "text": "child: hello\n"}]
    assert agent.messages[2]["extra"]["model_name"] == "child"
    assert "model_name" not in agent.messages[2]
    assert agent.cost == 1.0
    assert json.loads((tmp_path / "run.traj.json").read_text())["messages"] == agent.messages


@pytest.mark.parametrize(("sequence", "expected"), [(None, ["A", "B", "A"]), ([1, 0, 1], ["B", "A", "B"])])
def test_interleaving_observations_use_the_queried_child(sequence, expected, tmp_path, reset_global_stats):
    model = InterleavingModel(
        sequence=sequence,
        model_kwargs=[
            {
                "model_class": "deterministic",
                "model_name": name,
                "outputs": [make_output("run command", [{"command": "echo routed"}]) for _ in range(2)],
                "observation_template": name + ": {{output.output}}",
            }
            for name in ["A", "B"]
        ],
    )
    agent = DefaultAgent(
        model,
        LocalEnvironment(cwd=str(tmp_path)),
        system_template="system",
        instance_template="{{task}}",
        step_limit=3,
        cost_limit=0,
    )

    assert agent.run("task")["exit_status"] == "LimitsExceeded"
    assert [agent.messages[i]["content"] for i in [3, 5, 7]] == [f"{name}: routed\n" for name in expected]
    assert [agent.messages[i]["extra"]["model_name"] for i in [2, 4, 6]] == expected
    assert all("model_name" not in message for message in agent.messages)

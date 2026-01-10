import json
import tempfile
from pathlib import Path

from minisweagent.agents.default import DefaultAgent
from minisweagent.environments.local import LocalEnvironment
from minisweagent.models.test_models import DeterministicModel


def test_agent_save_includes_class_names():
    """Test that agent.save includes the full class names with import paths."""
    import yaml

    config_path = Path("src/minisweagent/config/default.yaml")
    with open(config_path) as f:
        default_config = yaml.safe_load(f)["agent"]

    model = DeterministicModel(outputs=["echo 'test'"])
    env = LocalEnvironment()
    agent = DefaultAgent(model, env, **default_config)

    agent.add_messages({"role": "system", "content": "test system message"})
    agent.add_messages({"role": "user", "content": "test user message"})

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir) / "test_trajectory.json"

        agent.save(temp_path, {"info": {"exit_status": "Submitted", "submission": "test result"}})

        with temp_path.open() as f:
            saved_data = json.load(f)

        assert "info" in saved_data
        assert "config" in saved_data["info"]

        config = saved_data["info"]["config"]

        assert "agent_type" in config
        assert "model_type" in config
        assert "environment_type" in config

        assert config["agent_type"] == "minisweagent.agents.default.DefaultAgent"
        assert config["model_type"] == "minisweagent.models.test_models.DeterministicModel"
        assert config["environment_type"] == "minisweagent.environments.local.LocalEnvironment"

        assert saved_data["info"]["exit_status"] == "Submitted"
        assert saved_data["info"]["submission"] == "test result"
        assert saved_data["trajectory_format"] == "mini-swe-agent-1.1"


def test_agent_serialize():
    """Test that agent.serialize returns the correct structure."""
    import yaml

    config_path = Path("src/minisweagent/config/default.yaml")
    with open(config_path) as f:
        default_config = yaml.safe_load(f)["agent"]

    model = DeterministicModel(outputs=["echo 'test'"])
    env = LocalEnvironment()
    agent = DefaultAgent(model, env, **default_config)

    agent.add_messages({"role": "system", "content": "test system message"})
    agent.add_messages({"role": "user", "content": "test user message"})

    data = agent.serialize()

    assert "info" in data
    assert "config" in data["info"]
    assert "messages" in data

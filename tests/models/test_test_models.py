import logging
import time

import minisweagent.models
from minisweagent.models.test_models import DeterministicModel, DeterministicModelConfig


def test_basic_functionality_and_cost_tracking(reset_global_stats):
    """Test basic model functionality, cost tracking, and default configuration."""
    # Model outputs must include bash blocks to avoid FormatError from parse_action
    model = DeterministicModel(
        outputs=["```mswea_bash_command\necho hello\n```", "```mswea_bash_command\necho world\n```"]
    )

    # Test first call with defaults
    result = model.query([{"role": "user", "content": "test"}])
    assert result["content"] == "```mswea_bash_command\necho hello\n```"
    assert result["extra"]["actions"] == [{"command": "echo hello"}]
    assert minisweagent.models.GLOBAL_MODEL_STATS.n_calls == 1
    assert minisweagent.models.GLOBAL_MODEL_STATS.cost == 1.0

    # Test second call and sequential outputs
    result = model.query([{"role": "user", "content": "test"}])
    assert result["content"] == "```mswea_bash_command\necho world\n```"
    assert result["extra"]["actions"] == [{"command": "echo world"}]
    assert minisweagent.models.GLOBAL_MODEL_STATS.n_calls == 2
    assert minisweagent.models.GLOBAL_MODEL_STATS.cost == 2.0


def test_custom_cost_and_multiple_models(reset_global_stats):
    """Test custom cost configuration and global tracking across multiple models."""
    # Model outputs must include bash blocks to avoid FormatError from parse_action
    model1 = DeterministicModel(outputs=["```mswea_bash_command\necho r1\n```"], cost_per_call=2.5)
    model2 = DeterministicModel(outputs=["```mswea_bash_command\necho r2\n```"], cost_per_call=3.0)

    result1 = model1.query([{"role": "user", "content": "test"}])
    assert result1["content"] == "```mswea_bash_command\necho r1\n```"
    assert minisweagent.models.GLOBAL_MODEL_STATS.cost == 2.5

    result2 = model2.query([{"role": "user", "content": "test"}])
    assert result2["content"] == "```mswea_bash_command\necho r2\n```"
    assert minisweagent.models.GLOBAL_MODEL_STATS.cost == 5.5
    assert minisweagent.models.GLOBAL_MODEL_STATS.n_calls == 2


def test_config_dataclass():
    """Test DeterministicModelConfig with custom values."""
    config = DeterministicModelConfig(outputs=["Test"], model_name="custom", cost_per_call=5.0)

    assert config.cost_per_call == 5.0
    assert config.model_name == "custom"

    model = DeterministicModel(**config.__dict__)
    assert model.config.cost_per_call == 5.0


def test_sleep_and_warning_commands(caplog):
    """Test special /sleep and /warning command handling."""
    # Test sleep command - processes sleep then returns actual output (counts as 1 call)
    # Model outputs must include bash blocks to avoid FormatError from parse_action
    model = DeterministicModel(outputs=["/sleep0.1", "```mswea_bash_command\necho after_sleep\n```"])
    start_time = time.time()
    result = model.query([{"role": "user", "content": "test"}])
    assert result["content"] == "```mswea_bash_command\necho after_sleep\n```"
    assert time.time() - start_time >= 0.1

    # Test warning command - processes warning then returns actual output (counts as 1 call)
    model2 = DeterministicModel(outputs=["/warningTest message", "```mswea_bash_command\necho after_warning\n```"])
    with caplog.at_level(logging.WARNING):
        result2 = model2.query([{"role": "user", "content": "test"}])
        assert result2["content"] == "```mswea_bash_command\necho after_warning\n```"
    assert "Test message" in caplog.text

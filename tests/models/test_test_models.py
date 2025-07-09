import logging
import time

import microswea.models
from microswea.models.test_models import DeterministicModel, DeterministicModelConfig


def reset_globals():
    """Reset global counters."""
    microswea.models.GLOBAL_COST = 0.0
    microswea.models.GLOBAL_N_CALLS = 0


def test_basic_functionality_and_cost_tracking():
    """Test basic model functionality, cost tracking, and default configuration."""
    reset_globals()

    model = DeterministicModel(outputs=["Hello", "World"])

    # Test first call with defaults
    assert model.query([{"role": "user", "content": "test"}]) == "Hello"
    assert model.n_calls == 1
    assert model.cost == 1.0
    assert microswea.models.GLOBAL_N_CALLS == 1
    assert microswea.models.GLOBAL_COST == 1.0

    # Test second call and sequential outputs
    assert model.query([{"role": "user", "content": "test"}]) == "World"
    assert model.n_calls == 2
    assert model.cost == 2.0
    assert microswea.models.GLOBAL_N_CALLS == 2
    assert microswea.models.GLOBAL_COST == 2.0


def test_custom_cost_and_multiple_models():
    """Test custom cost configuration and global tracking across multiple models."""
    reset_globals()

    model1 = DeterministicModel(outputs=["Response1"], cost_per_call=2.5)
    model2 = DeterministicModel(outputs=["Response2"], cost_per_call=3.0)

    assert model1.query([{"role": "user", "content": "test"}]) == "Response1"
    assert model1.cost == 2.5
    assert microswea.models.GLOBAL_COST == 2.5

    assert model2.query([{"role": "user", "content": "test"}]) == "Response2"
    assert model2.cost == 3.0
    assert microswea.models.GLOBAL_COST == 5.5
    assert microswea.models.GLOBAL_N_CALLS == 2


def test_config_dataclass():
    """Test DeterministicModelConfig with custom values."""
    config = DeterministicModelConfig(outputs=["Test"], model_name="custom", cost_per_call=5.0)

    assert config.cost_per_call == 5.0
    assert config.model_name == "custom"

    model = DeterministicModel(**config.__dict__)
    assert model.config.cost_per_call == 5.0


def test_sleep_and_warning_commands(caplog):
    """Test special /sleep and /warning command handling."""
    reset_globals()

    # Test sleep command - it recursively calls query so counts as 2 calls
    model = DeterministicModel(outputs=["/sleep0.1", "After sleep"])
    start_time = time.time()
    assert model.query([{"role": "user", "content": "test"}]) == "After sleep"
    assert time.time() - start_time >= 0.1
    assert model.n_calls == 2  # Sleep causes recursive call

    # Test warning command - also recursive
    model2 = DeterministicModel(outputs=["/warningTest message", "After warning"])
    with caplog.at_level(logging.WARNING):
        assert model2.query([{"role": "user", "content": "test"}]) == "After warning"
    assert model2.n_calls == 2  # Warning also causes recursive call
    assert "Test message" in caplog.text

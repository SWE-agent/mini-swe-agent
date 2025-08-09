"""Test that LitellmModel handles unknown models gracefully."""

from unittest.mock import MagicMock, patch

from minisweagent.models.litellm_model import LitellmModel, _warned_models


def test_unknown_model_graceful_handling():
    """Test that unknown models work without crashing and show warnings."""
    # Clear any previous warnings
    _warned_models.clear()

    # Create a mock response that would normally fail cost calculation
    mock_response = MagicMock()
    mock_response.choices = [MagicMock()]
    mock_response.choices[0].message.content = "Test response"

    with patch("minisweagent.models.litellm_model.litellm.completion") as mock_completion:
        mock_completion.return_value = mock_response

        # This model doesn't exist in litellm's registry
        model = LitellmModel(model_name="unknown/test-model")

        # Should work without crashing
        result = model.query([{"role": "user", "content": "test"}])

        assert result["content"] == "Test response"
        assert model.n_calls == 1
        assert model.unknown_model_calls == 1
        assert model.cost == 0  # Cost should be 0 for unknown models

        # Should have warned about this model
        assert "unknown/test-model" in _warned_models


def test_known_model_still_tracks_cost():
    """Test that known models still track cost properly."""
    mock_response = MagicMock()
    mock_response.choices = [MagicMock()]
    mock_response.choices[0].message.content = "Test response"

    with patch("minisweagent.models.litellm_model.litellm.completion") as mock_completion:
        mock_completion.return_value = mock_response

        with patch("minisweagent.models.litellm_model.litellm.cost_calculator.completion_cost") as mock_cost:
            mock_cost.return_value = 0.001  # $0.001

            model = LitellmModel(model_name="gpt-3.5-turbo")
            result = model.query([{"role": "user", "content": "test"}])

            assert result["content"] == "Test response"
            assert model.n_calls == 1
            assert model.unknown_model_calls == 0  # Should not increment for known models
            assert model.cost == 0.001  # Cost should be tracked


def test_warning_only_shown_once_per_model():
    """Test that warning is only shown once per model."""
    _warned_models.clear()

    mock_response = MagicMock()
    mock_response.choices = [MagicMock()]
    mock_response.choices[0].message.content = "Test response"

    with patch("minisweagent.models.litellm_model.litellm.completion") as mock_completion:
        mock_completion.return_value = mock_response

        model = LitellmModel(model_name="unknown/test-model-2")

        # First call should add to warned models
        model.query([{"role": "user", "content": "test"}])
        assert "unknown/test-model-2" in _warned_models

        # Second call should not warn again (just checking it doesn't crash)
        model.query([{"role": "user", "content": "test"}])
        assert model.unknown_model_calls == 2

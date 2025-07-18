from unittest.mock import Mock, patch

import litellm
import pytest

from microsweagent.models.litellm_model import LitellmModel


def test_authentication_error_enhanced_message():
    """Test that AuthenticationError gets enhanced with set-key instruction."""
    model = LitellmModel(model_name="gpt-4")

    # Create a mock exception that behaves like AuthenticationError
    original_error = Mock(spec=litellm.exceptions.AuthenticationError)
    original_error.message = "Invalid API key"

    with patch("litellm.completion") as mock_completion:
        # Make completion raise the mock error
        def side_effect(*args, **kwargs):
            raise litellm.exceptions.AuthenticationError("Invalid API key", llm_provider="openai", model="gpt-4")

        mock_completion.side_effect = side_effect

        with pytest.raises(litellm.exceptions.AuthenticationError) as exc_info:
            model._query([{"role": "user", "content": "test"}])

        # Check that the error message was enhanced
        assert "You can permanently set your API key with `micro-extra set-key KEY VALUE`." in str(exc_info.value)

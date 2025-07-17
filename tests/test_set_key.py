import sys
from pathlib import Path
from unittest.mock import patch

import pytest

from microsweagent.run.micro_extra import main


def test_set_key_success():
    """Test successful key setting with valid arguments."""
    test_args = ["micro", "set-key", "OPENAI_API_KEY", "sk-test123"]
    test_config_file = Path("/fake/config/.env")

    with (
        patch.object(sys, "argv", test_args),
        patch("microsweagent.run.micro_extra.global_config_file", test_config_file),
        patch("microsweagent.run.micro_extra.set_key") as mock_set_key,
    ):
        main()
        mock_set_key.assert_called_once_with(test_config_file, "OPENAI_API_KEY", "sk-test123")


def test_set_key_insufficient_args():
    """Test that ValueError is raised when insufficient arguments are provided."""
    test_args = ["micro", "set-key", "OPENAI_API_KEY"]  # Missing value

    with patch.object(sys, "argv", test_args):
        with pytest.raises(ValueError, match="Usage: micro set-key <key> <value>"):
            main()


def test_set_key_no_value_arg():
    """Test that ValueError is raised when no arguments are provided to set-key."""
    test_args = ["micro", "set-key"]  # Missing key and value

    with patch.object(sys, "argv", test_args):
        with pytest.raises(ValueError, match="Usage: micro set-key <key> <value>"):
            main()


def test_set_key_too_many_args():
    """Test that ValueError is raised when too many arguments are provided."""
    test_args = ["micro", "set-key", "OPENAI_API_KEY", "sk-test123", "extra"]

    with patch.object(sys, "argv", test_args):
        with pytest.raises(ValueError, match="Usage: micro set-key <key> <value>"):
            main()


def test_set_key_with_special_characters():
    """Test key setting with special characters in key and value."""
    test_args = ["micro", "set-key", "MY_API_KEY", "sk-proj-abc123!@#$%^&*()"]
    test_config_file = Path("/fake/config/.env")

    with (
        patch.object(sys, "argv", test_args),
        patch("microsweagent.run.micro_extra.global_config_file", test_config_file),
        patch("microsweagent.run.micro_extra.set_key") as mock_set_key,
    ):
        main()
        mock_set_key.assert_called_once_with(test_config_file, "MY_API_KEY", "sk-proj-abc123!@#$%^&*()")


def test_set_key_preserves_other_commands():
    """Test that other commands still work when set-key is not called."""
    test_args = ["micro", "gh", "issue-url"]

    with (
        patch.object(sys, "argv", test_args),
        patch("microsweagent.run.github_issue.app") as mock_gh_app,
    ):
        main()
        mock_gh_app.assert_called_once()


def test_set_key_does_not_interfere_with_inspector():
    """Test that inspector command still works when set-key is available."""
    test_args = ["micro", "i", "trajectory.json"]

    with (
        patch.object(sys, "argv", test_args),
        patch("microsweagent.run.inspector.app") as mock_inspect_app,
    ):
        main()
        mock_inspect_app.assert_called_once()

import os
from unittest.mock import patch

from minisweagent.run.extra.config import app, configure_if_first_time, edit, set, setup, unset


class TestConfigSetup:
    """Test the setup function with various inputs."""

    def test_setup_with_all_inputs(self, tmp_path):
        """Test setup function when user provides all inputs."""
        config_file = tmp_path / ".env"

        with (
            patch("minisweagent.run.extra.config.global_config_file", config_file),
            patch("minisweagent.run.extra.config.prompt") as mock_prompt,
            patch("minisweagent.run.extra.config.console.print"),
        ):
            mock_prompt.side_effect = ["anthropic/claude-sonnet-4-5-20250929", "ANTHROPIC_API_KEY", "sk-test123"]

            setup()

            # Verify the file was created and contains the expected content
            assert config_file.exists()
            content = config_file.read_text()
            assert "MSWEA_MODEL_NAME='anthropic/claude-sonnet-4-5-20250929'" in content
            assert "ANTHROPIC_API_KEY='sk-test123'" in content
            assert "MSWEA_CONFIGURED='true'" in content

    def test_setup_with_model_only(self, tmp_path):
        """Test setup when user only provides model name."""
        config_file = tmp_path / ".env"

        with (
            patch("minisweagent.run.extra.config.global_config_file", config_file),
            patch("minisweagent.run.extra.config.prompt") as mock_prompt,
            patch("minisweagent.run.extra.config.console.print"),
        ):
            mock_prompt.side_effect = ["gpt-4", "", ""]

            setup()

            content = config_file.read_text()
            assert "MSWEA_MODEL_NAME='gpt-4'" in content
            assert "MSWEA_CONFIGURED='true'" in content
            # Should not contain any API key
            assert "ANTHROPIC_API_KEY" not in content
            assert "OPENAI_API_KEY" not in content

    def test_setup_with_empty_inputs(self, tmp_path):
        """Test setup when user provides empty inputs."""
        config_file = tmp_path / ".env"

        with (
            patch("minisweagent.run.extra.config.global_config_file", config_file),
            patch("minisweagent.run.extra.config.prompt") as mock_prompt,
            patch("minisweagent.run.extra.config.console.print"),
        ):
            mock_prompt.side_effect = ["", "", ""]

            setup()

            content = config_file.read_text()
            # Should only have configured flag
            assert "MSWEA_CONFIGURED='true'" in content
            assert "MSWEA_MODEL_NAME" not in content

    def test_setup_with_existing_env_vars(self, tmp_path):
        """Test setup when environment variables already exist."""
        config_file = tmp_path / ".env"

        with (
            patch("minisweagent.run.extra.config.global_config_file", config_file),
            patch("minisweagent.run.extra.config.prompt") as mock_prompt,
            patch("minisweagent.run.extra.config.console.print"),
            patch.dict(os.environ, {"MSWEA_MODEL_NAME": "existing-model", "ANTHROPIC_API_KEY": "existing-key"}),
        ):
            # When prompted, user accepts defaults (existing values)
            mock_prompt.side_effect = ["existing-model", "ANTHROPIC_API_KEY", "existing-key"]

            setup()

            content = config_file.read_text()
            assert "MSWEA_MODEL_NAME='existing-model'" in content
            assert "ANTHROPIC_API_KEY='existing-key'" in content

    def test_setup_key_name_but_no_value(self, tmp_path):
        """Test setup when user provides key name but no value."""
        config_file = tmp_path / ".env"

        with (
            patch("minisweagent.run.extra.config.global_config_file", config_file),
            patch("minisweagent.run.extra.config.prompt") as mock_prompt,
            patch("minisweagent.run.extra.config.console.print") as mock_print,
        ):
            mock_prompt.side_effect = ["gpt-4", "OPENAI_API_KEY", ""]

            setup()

            content = config_file.read_text()
            assert "MSWEA_MODEL_NAME='gpt-4'" in content
            assert "MSWEA_CONFIGURED='true'" in content
            # Should not contain the API key since no value was provided
            assert "OPENAI_API_KEY" not in content
            mock_print.assert_any_call(
                "[bold red]API key setup not completed.[/bold red] Totally fine if you have your keys as environment variables."
            )


class TestConfigSet:
    """Test the set function for setting individual key-value pairs."""

    def test_set_with_both_arguments_provided(self, tmp_path):
        """Test set command when both key and value are provided as arguments."""
        config_file = tmp_path / ".env"

        with patch("minisweagent.run.extra.config.global_config_file", config_file):
            set("MSWEA_MODEL_NAME", "anthropic/claude-sonnet-4-5-20250929")

            assert config_file.exists()
            content = config_file.read_text()
            assert "MSWEA_MODEL_NAME='anthropic/claude-sonnet-4-5-20250929'" in content

    def test_set_with_no_arguments_prompts_for_both(self, tmp_path):
        """Test set command when no arguments provided - should prompt for both key and value."""
        config_file = tmp_path / ".env"

        with (
            patch("minisweagent.run.extra.config.global_config_file", config_file),
            patch("minisweagent.run.extra.config.prompt") as mock_prompt,
        ):
            mock_prompt.side_effect = ["TEST_KEY", "test_value"]

            set(None, None)

            assert mock_prompt.call_count == 2
            mock_prompt.assert_any_call("Enter the key to set: ")
            mock_prompt.assert_any_call("Enter the value for TEST_KEY: ")

            content = config_file.read_text()
            assert "TEST_KEY='test_value'" in content

    def test_set_with_key_only_prompts_for_value(self, tmp_path):
        """Test set command when only key is provided - should prompt for value."""
        config_file = tmp_path / ".env"

        with (
            patch("minisweagent.run.extra.config.global_config_file", config_file),
            patch("minisweagent.run.extra.config.prompt") as mock_prompt,
        ):
            mock_prompt.return_value = "prompted_value"

            set("PROVIDED_KEY", None)

            mock_prompt.assert_called_once_with("Enter the value for PROVIDED_KEY: ")

            content = config_file.read_text()
            assert "PROVIDED_KEY='prompted_value'" in content

    def test_set_with_value_only_prompts_for_key(self, tmp_path):
        """Test set command when only value is provided - should prompt for key."""
        config_file = tmp_path / ".env"

        with (
            patch("minisweagent.run.extra.config.global_config_file", config_file),
            patch("minisweagent.run.extra.config.prompt") as mock_prompt,
        ):
            mock_prompt.return_value = "prompted_key"

            set(None, "PROVIDED_VALUE")

            mock_prompt.assert_called_once_with("Enter the key to set: ")

            content = config_file.read_text()
            assert "prompted_key='PROVIDED_VALUE'" in content

    def test_set_key_value(self, tmp_path):
        """Test setting a key-value pair (legacy test for compatibility)."""
        config_file = tmp_path / ".env"

        with patch("minisweagent.run.extra.config.global_config_file", config_file):
            set("MSWEA_MODEL_NAME", "anthropic/claude-sonnet-4-5-20250929")

            assert config_file.exists()
            content = config_file.read_text()
            assert "MSWEA_MODEL_NAME='anthropic/claude-sonnet-4-5-20250929'" in content

    def test_set_api_key(self, tmp_path):
        """Test setting an API key."""
        config_file = tmp_path / ".env"

        with patch("minisweagent.run.extra.config.global_config_file", config_file):
            set("ANTHROPIC_API_KEY", "sk-anthropic-test-key")

            content = config_file.read_text()
            assert "ANTHROPIC_API_KEY='sk-anthropic-test-key'" in content

    def test_set_multiple_keys(self, tmp_path):
        """Test setting multiple keys in sequence."""
        config_file = tmp_path / ".env"

        with patch("minisweagent.run.extra.config.global_config_file", config_file):
            set("MSWEA_MODEL_NAME", "gpt-4")
            set("OPENAI_API_KEY", "sk-openai-test")
            set("MSWEA_GLOBAL_COST_LIMIT", "10.00")

            content = config_file.read_text()
            assert "MSWEA_MODEL_NAME='gpt-4'" in content
            assert "OPENAI_API_KEY='sk-openai-test'" in content
            assert "MSWEA_GLOBAL_COST_LIMIT='10.00'" in content

    def test_set_overwrites_existing_key(self, tmp_path):
        """Test that setting a key overwrites existing value."""
        config_file = tmp_path / ".env"
        config_file.write_text("MSWEA_MODEL_NAME=old-model\nOTHER_KEY=other-value\n")

        with patch("minisweagent.run.extra.config.global_config_file", config_file):
            set("MSWEA_MODEL_NAME", "new-model")

            content = config_file.read_text()
            assert "MSWEA_MODEL_NAME='new-model'" in content
            assert "old-model" not in content
            # Other keys should remain unchanged
            assert "OTHER_KEY=other-value" in content

    def test_set_with_empty_strings_via_prompt(self, tmp_path):
        """Test set command when prompted values are empty strings."""
        config_file = tmp_path / ".env"

        with (
            patch("minisweagent.run.extra.config.global_config_file", config_file),
            patch("minisweagent.run.extra.config.prompt") as mock_prompt,
        ):
            mock_prompt.side_effect = ["EMPTY_KEY", ""]

            set(None, None)

            content = config_file.read_text()
            assert "EMPTY_KEY=''" in content


class TestConfigUnset:
    """Test the unset function for removing key-value pairs."""

    def test_unset_with_argument_provided(self, tmp_path):
        """Test unset command when key is provided as argument."""
        config_file = tmp_path / ".env"
        config_file.write_text("MSWEA_MODEL_NAME='gpt-4'\nOPENAI_API_KEY='sk-test123'\n")

        with patch("minisweagent.run.extra.config.global_config_file", config_file):
            unset("MSWEA_MODEL_NAME")

            content = config_file.read_text()
            assert "MSWEA_MODEL_NAME" not in content
            # Other keys should remain
            assert "OPENAI_API_KEY='sk-test123'" in content

    def test_unset_with_no_argument_prompts_for_key(self, tmp_path):
        """Test unset command when no argument provided - should prompt for key."""
        config_file = tmp_path / ".env"
        config_file.write_text("TEST_KEY='test_value'\nOTHER_KEY='other_value'\n")

        with (
            patch("minisweagent.run.extra.config.global_config_file", config_file),
            patch("minisweagent.run.extra.config.prompt") as mock_prompt,
        ):
            mock_prompt.return_value = "TEST_KEY"

            unset(None)

            mock_prompt.assert_called_once_with("Enter the key to unset: ")

            content = config_file.read_text()
            assert "TEST_KEY" not in content
            assert "OTHER_KEY='other_value'" in content

    def test_unset_existing_key(self, tmp_path):
        """Test unsetting an existing key (legacy test for compatibility)."""
        config_file = tmp_path / ".env"
        config_file.write_text("MSWEA_MODEL_NAME='gpt-4'\nOPENAI_API_KEY='sk-test123'\n")

        with patch("minisweagent.run.extra.config.global_config_file", config_file):
            unset("MSWEA_MODEL_NAME")

            content = config_file.read_text()
            assert "MSWEA_MODEL_NAME" not in content
            # Other keys should remain
            assert "OPENAI_API_KEY='sk-test123'" in content

    def test_unset_nonexistent_key(self, tmp_path):
        """Test unsetting a key that doesn't exist (should not error)."""
        config_file = tmp_path / ".env"
        config_file.write_text("MSWEA_MODEL_NAME='gpt-4'\n")

        with patch("minisweagent.run.extra.config.global_config_file", config_file):
            # Should not raise an exception
            unset("NONEXISTENT_KEY")

            content = config_file.read_text()
            # Original content should remain unchanged
            assert "MSWEA_MODEL_NAME='gpt-4'" in content

    def test_unset_from_empty_file(self, tmp_path):
        """Test unsetting from an empty file."""
        config_file = tmp_path / ".env"
        config_file.write_text("")

        with patch("minisweagent.run.extra.config.global_config_file", config_file):
            # Should not raise an exception
            unset("ANY_KEY")

            # File should remain empty
            content = config_file.read_text()
            assert content == ""

    def test_unset_from_file_with_multiple_keys(self, tmp_path):
        """Test unsetting one key from a file with multiple keys."""
        config_file = tmp_path / ".env"
        config_file.write_text(
            "MSWEA_MODEL_NAME='anthropic/claude-sonnet-4-5-20250929'\n"
            "ANTHROPIC_API_KEY='sk-anthropic-key'\n"
            "OPENAI_API_KEY='sk-openai-key'\n"
            "MSWEA_CONFIGURED='true'\n"
        )

        with patch("minisweagent.run.extra.config.global_config_file", config_file):
            unset("ANTHROPIC_API_KEY")

            content = config_file.read_text()
            # Target key should be removed
            assert "ANTHROPIC_API_KEY" not in content
            # Other keys should remain
            assert "MSWEA_MODEL_NAME='anthropic/claude-sonnet-4-5-20250929'" in content
            assert "OPENAI_API_KEY='sk-openai-key'" in content
            assert "MSWEA_CONFIGURED='true'" in content

    def test_unset_api_key_scenario(self, tmp_path):
        """Test unsetting an API key specifically."""
        config_file = tmp_path / ".env"
        config_file.write_text("MSWEA_MODEL_NAME='gpt-4'\nOPENAI_API_KEY='sk-old-key'\nMSWEA_CONFIGURED='true'\n")

        with patch("minisweagent.run.extra.config.global_config_file", config_file):
            unset("OPENAI_API_KEY")

            content = config_file.read_text()
            # API key should be completely removed
            assert "OPENAI_API_KEY" not in content
            assert "sk-old-key" not in content
            # Other config should remain
            assert "MSWEA_MODEL_NAME='gpt-4'" in content
            assert "MSWEA_CONFIGURED='true'" in content

    def test_unset_configured_flag(self, tmp_path):
        """Test unsetting the configured flag."""
        config_file = tmp_path / ".env"
        config_file.write_text("MSWEA_MODEL_NAME='gpt-4'\nMSWEA_CONFIGURED='true'\n")

        with patch("minisweagent.run.extra.config.global_config_file", config_file):
            unset("MSWEA_CONFIGURED")

            content = config_file.read_text()
            # Configured flag should be removed
            assert "MSWEA_CONFIGURED" not in content
            # Model should remain
            assert "MSWEA_MODEL_NAME='gpt-4'" in content


class TestConfigEdit:
    """Test the edit function."""

    def test_edit_with_default_editor(self, tmp_path):
        """Test edit function with default editor (nano)."""
        config_file = tmp_path / ".env"
        config_file.write_text("MSWEA_MODEL_NAME=test")

        with (
            patch("minisweagent.run.extra.config.global_config_file", config_file),
            patch("subprocess.run") as mock_run,
            patch.dict(os.environ, {}, clear=True),  # Clear EDITOR env var
        ):
            edit()

            mock_run.assert_called_once_with(["nano", config_file])

    def test_edit_with_custom_editor(self, tmp_path):
        """Test edit function with custom editor."""
        config_file = tmp_path / ".env"
        config_file.write_text("MSWEA_MODEL_NAME=test")

        with (
            patch("minisweagent.run.extra.config.global_config_file", config_file),
            patch("subprocess.run") as mock_run,
            patch.dict(os.environ, {"EDITOR": "vim"}),
        ):
            edit()

            mock_run.assert_called_once_with(["vim", config_file])


class TestConfigureIfFirstTime:
    """Test the configure_if_first_time function."""

    def test_configure_when_not_configured(self, tmp_path):
        """Test that setup is called when MSWEA_CONFIGURED is not set."""
        config_file = tmp_path / ".env"

        with (
            patch("minisweagent.run.extra.config.global_config_file", config_file),
            patch("minisweagent.run.extra.config.setup") as mock_setup,
            patch("minisweagent.run.extra.config.console.print") as mock_print,
            patch.dict(os.environ, {}, clear=True),  # Clear MSWEA_CONFIGURED
        ):
            configure_if_first_time()

            mock_setup.assert_called_once()
            mock_print.assert_called()

    def test_skip_configure_when_already_configured(self, tmp_path):
        """Test that setup is not called when MSWEA_CONFIGURED is set."""
        with (
            patch("minisweagent.run.extra.config.setup") as mock_setup,
            patch.dict(os.environ, {"MSWEA_CONFIGURED": "true"}),
        ):
            configure_if_first_time()

            mock_setup.assert_not_called()


class TestTyperAppIntegration:
    """Test the Typer app commands directly."""

    def test_set_command_via_typer(self, tmp_path):
        """Test the set command through the Typer app."""
        config_file = tmp_path / ".env"

        with (
            patch("minisweagent.run.extra.config.global_config_file", config_file),
            patch("typer.Option") as mock_option,
        ):
            # Mock the typer Option to return our test values
            mock_option.side_effect = (
                lambda default, **kwargs: "OPENAI_API_KEY" if "key" in str(kwargs) else "sk-test-key"
            )

            # Call the set function directly (as the app would)
            set("OPENAI_API_KEY", "sk-test-key")

            content = config_file.read_text()
            assert "OPENAI_API_KEY='sk-test-key'" in content

    def test_unset_command_via_typer(self, tmp_path):
        """Test the unset command through the Typer app."""
        config_file = tmp_path / ".env"
        config_file.write_text("OPENAI_API_KEY='sk-test-key'\nMSWEA_MODEL_NAME='gpt-4'\n")

        with patch("minisweagent.run.extra.config.global_config_file", config_file):
            # Call the unset function directly (as the app would)
            unset("OPENAI_API_KEY")

            content = config_file.read_text()
            assert "OPENAI_API_KEY" not in content
            assert "MSWEA_MODEL_NAME='gpt-4'" in content

    def test_app_help_contains_config_file_path(self):
        """Test that the app help string includes the config file path."""
        help_text = app.info.help
        assert help_text is not None
        assert "global_config_file" in help_text or ".env" in help_text

    def test_app_no_args_is_help(self):
        """Test that the app shows help when no arguments provided."""
        assert app.info.no_args_is_help is True

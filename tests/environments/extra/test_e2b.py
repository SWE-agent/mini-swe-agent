"""Tests for the E2B cloud sandbox environment."""

from types import ModuleType
from unittest.mock import MagicMock, patch

import pytest

from minisweagent.environments.extra.e2b import (
    E2BEnvironment,
    E2BEnvironmentConfig,
    E2BTemplateManager,
)
from minisweagent.exceptions import Submitted

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_mock_e2b() -> ModuleType:
    """Return a minimal mock of the `e2b` module."""
    mock_e2b = MagicMock()
    mock_e2b.Template = MagicMock()
    mock_e2b.Sandbox = MagicMock()
    return mock_e2b


def _make_env(**kwargs) -> E2BEnvironment:
    """Create an E2BEnvironment without touching real E2B infrastructure."""
    with patch.object(E2BEnvironment, "__init__", lambda self, **kw: None):
        env = E2BEnvironment()
        env.config = E2BEnvironmentConfig(image="swebench/test-image:latest", **kwargs)
        env.sandbox = MagicMock()
        env.logger = MagicMock()
        return env


# ---------------------------------------------------------------------------
# E2BEnvironmentConfig
# ---------------------------------------------------------------------------


class TestE2BEnvironmentConfig:
    def test_defaults(self):
        cfg = E2BEnvironmentConfig(image="python:3.11")
        assert cfg.cwd == "/"
        assert cfg.timeout == 30
        assert cfg.sandbox_timeout == 3600
        assert cfg.cpu_count == 2
        assert cfg.memory_mb == 2048
        assert cfg.skip_cache is False
        assert cfg.tags == []
        assert cfg.build_timeout == 1800
        assert cfg.api_key is None
        assert cfg.registry_username is None
        assert cfg.registry_password is None

    def test_custom_values(self):
        cfg = E2BEnvironmentConfig(image="my-image:tag", sandbox_timeout=7200, cpu_count=4)
        assert cfg.sandbox_timeout == 7200
        assert cfg.cpu_count == 4


# ---------------------------------------------------------------------------
# E2BTemplateManager._image_to_template_name
# ---------------------------------------------------------------------------


class TestImageToTemplateName:
    def test_basic_sanitization(self):
        name = E2BTemplateManager._image_to_template_name("python:3.11")
        assert re.match(r"^[a-z0-9-]+$", name), f"Invalid chars in: {name}"

    def test_length_limit(self):
        long_image = "a" * 100 + ":latest"
        name = E2BTemplateManager._image_to_template_name(long_image)
        assert len(name) <= 63

    def test_deterministic(self):
        image = "swebench/sweb.eval.x86_64.django__django-11099:latest"
        assert E2BTemplateManager._image_to_template_name(image) == E2BTemplateManager._image_to_template_name(image)

    def test_different_images_different_names(self):
        a = E2BTemplateManager._image_to_template_name("image-a:latest")
        b = E2BTemplateManager._image_to_template_name("image-b:latest")
        assert a != b

    def test_no_triple_hyphens(self):
        # Dots and slashes become hyphens; consecutive runs are collapsed to "--"
        name = E2BTemplateManager._image_to_template_name("a/b/c.d.e:latest")
        assert "---" not in name

    def test_empty_prefix_falls_back_to_hash(self):
        # An image that sanitizes to only hyphens should return just the hash
        name = E2BTemplateManager._image_to_template_name("---")
        assert len(name) == 8  # just the 8-char sha256 prefix


import re  # noqa: E402 (needed after class definitions above for clarity)

# ---------------------------------------------------------------------------
# E2BEnvironment.execute
# ---------------------------------------------------------------------------


class TestE2BEnvironmentExecute:
    def test_execute_dict_action(self):
        env = _make_env()
        mock_result = MagicMock()
        mock_result.stdout = "hello\n"
        mock_result.stderr = ""
        mock_result.exit_code = 0
        env.sandbox.commands.run.return_value = mock_result

        output = env.execute({"command": "echo hello"})

        assert output["output"] == "hello\n"
        assert output["returncode"] == 0
        assert output["exception_info"] == ""

    def test_execute_string_action(self):
        env = _make_env()
        mock_result = MagicMock()
        mock_result.stdout = "ok\n"
        mock_result.stderr = ""
        mock_result.exit_code = 0
        env.sandbox.commands.run.return_value = mock_result

        output = env.execute("echo ok")

        assert output["output"] == "ok\n"

    def test_execute_nonzero_exit(self):
        env = _make_env()
        mock_result = MagicMock()
        mock_result.stdout = ""
        mock_result.stderr = "error\n"
        mock_result.exit_code = 1
        env.sandbox.commands.run.return_value = mock_result

        output = env.execute({"command": "false"})

        assert output["returncode"] == 1

    def test_execute_exception(self):
        env = _make_env()
        env.sandbox.commands.run.side_effect = RuntimeError("connection lost")

        output = env.execute({"command": "ls"})

        assert output["returncode"] == -1
        assert "connection lost" in output["exception_info"]

    def test_execute_raises_submitted(self):
        env = _make_env()
        mock_result = MagicMock()
        mock_result.stdout = "COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT\ndiff --git a/f.py b/f.py\n"
        mock_result.stderr = ""
        mock_result.exit_code = 0
        env.sandbox.commands.run.return_value = mock_result

        with pytest.raises(Submitted) as exc_info:
            env.execute({"command": "submit"})

        msg = exc_info.value.messages[0]
        assert msg["extra"]["exit_status"] == "Submitted"
        assert "diff --git" in msg["extra"]["submission"]


# ---------------------------------------------------------------------------
# E2BEnvironment.serialize
# ---------------------------------------------------------------------------


class TestE2BEnvironmentSerialize:
    def test_serialize_structure(self):
        env = _make_env()
        result = env.serialize()

        assert "info" in result
        assert "config" in result["info"]
        assert "environment" in result["info"]["config"]
        assert "environment_type" in result["info"]["config"]
        assert "E2BEnvironment" in result["info"]["config"]["environment_type"]

    def test_serialize_excludes_credentials(self):
        env = _make_env()
        env.config.api_key = "secret-key"
        env.config.registry_password = "secret-pass"

        result = env.serialize()
        env_cfg = result["info"]["config"]["environment"]

        assert "api_key" not in env_cfg
        assert "registry_password" not in env_cfg


# ---------------------------------------------------------------------------
# E2BEnvironment.stop / __del__
# ---------------------------------------------------------------------------


class TestE2BEnvironmentStop:
    def test_stop_kills_sandbox(self):
        env = _make_env()
        env.stop()
        env.sandbox.kill.assert_called_once()

    def test_stop_tolerates_missing_sandbox(self):
        with patch.object(E2BEnvironment, "__init__", lambda self, **kw: None):
            env = E2BEnvironment()
            # sandbox was never set
            env.stop()  # should not raise

    def test_stop_tolerates_kill_exception(self):
        env = _make_env()
        env.sandbox.kill.side_effect = RuntimeError("already dead")
        env.stop()  # should not raise


# ---------------------------------------------------------------------------
# atexit cleanup registry
# ---------------------------------------------------------------------------


class TestAtexitCleanup:
    def test_stop_removes_from_active_sandboxes(self):
        from minisweagent.environments.extra import e2b as e2b_mod

        env = _make_env()
        e2b_mod._active_sandboxes.add(env)
        assert env in e2b_mod._active_sandboxes

        env.stop()
        assert env not in e2b_mod._active_sandboxes

    def test_cleanup_all_sandboxes_kills_all(self):
        from minisweagent.environments.extra import e2b as e2b_mod

        env1 = _make_env()
        env2 = _make_env()
        e2b_mod._active_sandboxes.update([env1, env2])

        e2b_mod._cleanup_all_sandboxes()

        env1.sandbox.kill.assert_called_once()
        env2.sandbox.kill.assert_called_once()
        assert env1 not in e2b_mod._active_sandboxes
        assert env2 not in e2b_mod._active_sandboxes

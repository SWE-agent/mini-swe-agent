"""Tests for minisweagent.__init__."""

import importlib.metadata
import os
import subprocess
import sys

from packaging.requirements import Requirement


def test_startup_banner_survives_non_utf8_stdout(tmp_path):
    """Importing the package must not crash when stdout can't encode the startup banner (e.g. Windows cp1252)."""
    env = {
        **os.environ,
        "PYTHONIOENCODING": "cp1252",
        "MSWEA_SILENT_STARTUP": "",
        "MSWEA_GLOBAL_CONFIG_DIR": str(tmp_path),
    }
    result = subprocess.run([sys.executable, "-c", "import minisweagent"], capture_output=True, text=True, env=env)
    assert result.returncode == 0, result.stderr


def test_litellm_dependency_is_pinned():
    requirements = [Requirement(req) for req in importlib.metadata.requires("mini-swe-agent") or []]
    litellm = next(req for req in requirements if req.name == "litellm")
    assert str(litellm.specifier) == "==1.96.0"

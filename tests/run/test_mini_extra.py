import os
import subprocess
import sys
from pathlib import Path

import pytest


def run_mini_extra(args: list[str], config_dir: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-m", "minisweagent.run.utilities.mini_extra", *args],
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=15,
        env=os.environ | {"MSWEA_GLOBAL_CONFIG_DIR": str(config_dir), "MSWEA_SILENT_STARTUP": "1", "NO_COLOR": "1"},
    )


@pytest.mark.parametrize(
    ("args", "expected_returncode"),
    [
        (["swe-bnech"], 2),
        (["--unknown-option"], 2),
        (["[bold]unknown[/bold]"], 2),
        ([], 0),
        (["-h"], 0),
        (["--help"], 0),
    ],
)
def test_mini_extra_exit_status(args: list[str], expected_returncode: int, tmp_path: Path):
    result = run_mini_extra(args, tmp_path)
    assert result.returncode == expected_returncode
    if expected_returncode:
        assert f"Unknown command: {args[0]}" in result.stderr
    else:
        assert "Available sub-commands:" in result.stdout
        assert not result.stderr


def test_known_command_still_dispatches(tmp_path: Path):
    result = run_mini_extra(["config", "--help"], tmp_path)
    assert result.returncode == 0
    assert "Usage: mini-extra config" in result.stdout
    assert not result.stderr

from contextlib import contextmanager
from pathlib import Path
from unittest.mock import patch

import pytest

from repopilot.runner.run_task import (
    build_mini_command,
    isolated_workspace,
    resolve_task_dir,
    run_benchmark_task,
)
from repopilot.schema import load_task

ROOT = Path(__file__).resolve().parents[2]
BENCHMARKS = ROOT / "benchmarks"
TASK_001 = BENCHMARKS / "task_001_sudoku"
TASK_002 = BENCHMARKS / "task_002_eval_module"


def test_resolve_task_dir_by_name():
    assert resolve_task_dir(Path("task_001_sudoku"), benchmarks_root=BENCHMARKS) == TASK_001.resolve()


def test_resolve_task_dir_by_path():
    assert resolve_task_dir(TASK_001) == TASK_001.resolve()


def test_resolve_task_dir_missing():
    with pytest.raises(FileNotFoundError):
        resolve_task_dir(Path("no_such_task"), benchmarks_root=BENCHMARKS)


@patch("repopilot.runner.run_task.shutil.which", return_value="/usr/bin/mini")
def test_build_mini_command(_which):
    task = load_task(TASK_001)
    cmd = build_mini_command(task, Path("runs/task_001_sudoku/trajectory.traj.json"))
    assert cmd[0] == "/usr/bin/mini"
    assert "-t" in cmd
    assert "upstream/tests/run/test_sudoku.py" in cmd[cmd.index("-t") + 1]
    assert cmd[-2:] == ["-o", "runs/task_001_sudoku/trajectory.traj.json"]
    assert "-y" in cmd
    assert "--exit-immediately" in cmd


@patch("repopilot.runner.run_task.shutil.which", return_value="/usr/bin/mini")
def test_run_benchmark_task_dry_run(_which, capsys):
    result = run_benchmark_task(TASK_001, project_root=ROOT, dry_run=True)
    assert result.task_id == "task_001_sudoku"
    assert result.mini_exit_code is None
    assert "/usr/bin/mini" in result.mini_command[0]
    captured = capsys.readouterr()
    assert "task_001_sudoku" in captured.out
    assert "git worktree" in captured.out


@contextmanager
def _noop_workspace(*_args, **kwargs):
    yield kwargs.get("worktree_path", ROOT)


@patch("repopilot.runner.run_task.run_verification", return_value=1)
@patch("repopilot.runner.run_task.isolated_workspace")
@patch("repopilot.runner.run_task.shutil.which", return_value="/usr/bin/mini")
def test_skip_mini_skips_agent_and_records_verify_failure(_which, mock_workspace, _verify):
    mock_workspace.side_effect = lambda *args, **kwargs: _noop_workspace(*args, **kwargs)
    result = run_benchmark_task(
        TASK_001,
        project_root=ROOT,
        skip_mini=True,
        restore_workspace=True,
    )
    assert result.mini_exit_code is None
    assert result.tests_passed is False
    assert result.test_exit_code == 1
    assert (result.output_dir / "verify_test.log").is_file()
    assert (result.output_dir / "run_meta.yaml").is_file()


@pytest.mark.skipif(not TASK_002.is_dir(), reason="requires task_002_eval_module")
def test_isolated_workspace_does_not_change_dev_branch():
    import subprocess

    before = subprocess.run(
        ["git", "rev-parse", "--abbrev-ref", "HEAD"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    worktree = ROOT / "runs" / "_isolation_test" / ".workspace"
    with isolated_workspace(
        ROOT,
        "61e468b",
        TASK_002 / "setup.patch",
        worktree_path=worktree,
    ) as workspace:
        assert workspace.is_dir()
        assert (workspace / "benchmarks" / "task_001_sudoku").is_dir()
        assert (workspace / "upstream" / "src" / "minisweagent" / "run" / "eval_module.py").is_file()
        assert (workspace / "upstream" / "src" / "minisweagent" / "run" / "expr" / "evaluate.py").is_file()
    after = subprocess.run(
        ["git", "rev-parse", "--abbrev-ref", "HEAD"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    assert before == after
    assert not worktree.exists()


@pytest.mark.skipif(not TASK_002.is_dir(), reason="requires task_002_eval_module")
def test_setup_patch_stages_buggy_state_in_index():
    import subprocess

    worktree = ROOT / "runs" / "_index_test" / ".workspace"
    try:
        with isolated_workspace(
            ROOT,
            "61e468b",
            TASK_002 / "setup.patch",
            worktree_path=worktree,
        ):
            wt_vs_index = subprocess.run(
                ["git", "diff", "--stat"],
                cwd=worktree,
                capture_output=True,
                text=True,
                check=True,
            ).stdout.strip()
            cached = subprocess.run(
                ["git", "diff", "--cached", "--stat"],
                cwd=worktree,
                capture_output=True,
                text=True,
                check=True,
            ).stdout.strip()
            assert wt_vs_index == ""
            assert "eval_module.py" in cached
    finally:
        if worktree.is_dir():
            subprocess.run(
                ["git", "worktree", "remove", "--force", str(worktree)],
                cwd=ROOT,
                capture_output=True,
            )

"""Benchmark task orchestration (Phase 1.5)."""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import yaml

from repopilot.schema import TaskConfig, load_task
from repopilot.trace import TraceContext, record_trace
from repopilot.trace.parse import capture_workspace_diff


@dataclass
class RunResult:
    task_id: str
    base_commit: str
    agent_mode: str
    repo_path: Path
    output_dir: Path
    trajectory_path: Path
    mini_exit_code: int | None
    test_exit_code: int
    tests_passed: bool
    mini_command: list[str]


def find_project_root(start: Path | None = None) -> Path:
    start = (start or Path.cwd()).resolve()
    result = subprocess.run(
        ["git", "rev-parse", "--show-toplevel"],
        cwd=start,
        capture_output=True,
        text=True,
        check=True,
    )
    return Path(result.stdout.strip())


def resolve_task_dir(task_arg: Path, *, benchmarks_root: Path | None = None) -> Path:
    task_arg = task_arg.expanduser()
    if task_arg.is_dir():
        return task_arg.resolve()

    root = find_project_root()
    benchmarks_root = (benchmarks_root or root / "benchmarks").resolve()
    for candidate in (task_arg, benchmarks_root / task_arg, benchmarks_root / task_arg.name):
        if candidate.is_dir():
            return candidate.resolve()
    raise FileNotFoundError(f"Benchmark task directory not found: {task_arg}")


def resolve_repo_path(task: TaskConfig, project_root: Path) -> Path:
    if task.repo.path is not None:
        repo = Path(task.repo.path)
        return repo.resolve() if repo.is_absolute() else (project_root / repo).resolve()
    if task.repo.repo_url is None:
        raise ValueError("repo.path or repo.repo_url must be set")
    raise NotImplementedError("Cloning repo_url is not implemented yet (Phase 1.5 supports repo.path only)")


def build_mini_command(task: TaskConfig, trajectory_path: Path) -> list[str]:
    mini = shutil.which("mini")
    if mini is None:
        raise RuntimeError("mini not found on PATH; install upstream with: pip install -e 'upstream/[dev]'")
    return [
        mini,
        *task.agent.mini_flags,
        "-t",
        task.read_issue(),
        "-o",
        str(trajectory_path),
    ]


def _remove_worktree(repo_path: Path, worktree_path: Path) -> None:
    if not worktree_path.is_dir():
        return
    result = subprocess.run(
        ["git", "worktree", "remove", "--force", str(worktree_path)],
        cwd=repo_path,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0 and worktree_path.exists():
        shutil.rmtree(worktree_path, ignore_errors=True)
        subprocess.run(["git", "worktree", "prune"], cwd=repo_path, check=False)


@contextmanager
def isolated_workspace(
    repo_path: Path,
    base_commit: str,
    setup_patch: Path | None,
    *,
    worktree_path: Path,
    keep: bool = False,
):
    """Prepare a detached git worktree for the task without touching the dev checkout."""
    worktree_path = worktree_path.resolve()
    worktree_path.parent.mkdir(parents=True, exist_ok=True)
    _remove_worktree(repo_path, worktree_path)
    subprocess.run(
        ["git", "worktree", "add", "--detach", str(worktree_path), base_commit],
        cwd=repo_path,
        check=True,
    )
    if setup_patch is not None:
        subprocess.run(["git", "apply", "--index", str(setup_patch)], cwd=worktree_path, check=True)
    try:
        yield worktree_path
    finally:
        if not keep:
            _remove_worktree(repo_path, worktree_path)


# Backward-compatible alias for tests/mocks.
prepared_workspace = isolated_workspace


def _workspace_env(workspace: Path) -> dict[str, str]:
    env = os.environ.copy()
    upstream_src = workspace / "upstream" / "src"
    if upstream_src.is_dir():
        prefix = str(upstream_src)
        existing = env.get("PYTHONPATH")
        env["PYTHONPATH"] = f"{prefix}{os.pathsep}{existing}" if existing else prefix
    return env


def run_verification(task: TaskConfig, repo_path: Path, log_path: Path) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    result = subprocess.run(
        task.test_command,
        shell=True,
        cwd=repo_path,
        env=_workspace_env(repo_path),
        capture_output=True,
        text=True,
    )
    log_path.write_text(result.stdout + result.stderr)
    return result.returncode


def extract_baseline_artifacts(
    trajectory_path: Path,
    output_dir: Path,
    *,
    issue_path: Path | None = None,
    project_root: Path | None = None,
    workspace: Path | None = None,
) -> None:
    root = project_root or find_project_root()
    script = root / "scripts" / "extract_baseline.py"
    if not script.is_file():
        raise FileNotFoundError(f"Missing extraction script: {script}")

    cmd = [sys.executable, str(script), str(trajectory_path), "-o", str(output_dir)]
    if issue_path is not None:
        cmd.extend(["--issue", str(issue_path)])
    if workspace is not None:
        cmd.extend(["--workspace", str(workspace)])
    subprocess.run(cmd, check=True)


def write_run_meta(result: RunResult, *, started_at: datetime, finished_at: datetime) -> Path:
    meta_path = result.output_dir / "run_meta.yaml"
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "task_id": result.task_id,
        "base_commit": result.base_commit,
        "agent_mode": result.agent_mode,
        "repo_path": str(result.repo_path),
        "output_dir": str(result.output_dir),
        "trajectory": str(result.trajectory_path),
        "mini_command": result.mini_command,
        "mini_exit_code": result.mini_exit_code,
        "test_exit_code": result.test_exit_code,
        "tests_passed": result.tests_passed,
        "started_at": started_at.isoformat(),
        "finished_at": finished_at.isoformat(),
    }
    meta_path.write_text(yaml.safe_dump(payload, sort_keys=False))
    return meta_path


def run_benchmark_task(
    task_dir: Path,
    *,
    project_root: Path | None = None,
    skip_mini: bool = False,
    restore_workspace: bool = True,
    dry_run: bool = False,
    force_baseline: bool = False,
) -> RunResult:
    started_at = datetime.now(timezone.utc)
    task = load_task(task_dir)
    if force_baseline and task.agent.mode != "baseline":
        raise ValueError(f"Task {task.task_id} is not baseline mode")
    if task.agent.mode == "repopilot":
        raise NotImplementedError("repopilot agent mode is not implemented yet")

    root = (project_root or find_project_root(task_dir)).resolve()
    repo_path = resolve_repo_path(task, root)
    output_dir = (root / "runs" / task.task_id).resolve()
    trajectory_path = (root / task.agent.resolve_output_trajectory(task.task_id)).resolve()
    verify_log = output_dir / "verify_test.log"
    mini_command = build_mini_command(task, trajectory_path)

    workspace_path = output_dir / ".workspace"

    if dry_run:
        typer_echo = _get_typer_echo()
        typer_echo(f"task_id: {task.task_id}")
        typer_echo(f"repo_path: {repo_path}")
        typer_echo(f"workspace: {workspace_path} (git worktree at {task.repo.base_commit})")
        typer_echo(f"trajectory: {trajectory_path}")
        typer_echo(f"mini: {' '.join(mini_command)}")
        typer_echo(f"verify: {task.test_command}")
        return RunResult(
            task_id=task.task_id,
            base_commit=task.repo.base_commit,
            agent_mode=task.agent.mode,
            repo_path=repo_path,
            output_dir=output_dir,
            trajectory_path=trajectory_path,
            mini_exit_code=None,
            test_exit_code=0,
            tests_passed=False,
            mini_command=mini_command,
        )

    mini_exit_code: int | None = None
    with isolated_workspace(
        repo_path,
        task.repo.base_commit,
        task.setup_patch_path(),
        worktree_path=workspace_path,
        keep=not restore_workspace,
    ) as workspace:
        if not skip_mini:
            output_dir.mkdir(parents=True, exist_ok=True)
            mini_result = subprocess.run(mini_command, cwd=workspace, env=_workspace_env(workspace))
            mini_exit_code = mini_result.returncode

        test_exit_code = run_verification(task, workspace, verify_log)
        tests_passed = test_exit_code == 0
        workspace_patch = capture_workspace_diff(workspace)

        if trajectory_path.is_file():
            extract_baseline_artifacts(
                trajectory_path,
                output_dir,
                issue_path=task.issue_path(),
                project_root=root,
                workspace=workspace,
            )
            record_trace(
                trajectory_path,
                output_dir,
                ctx=TraceContext(
                    task_id=task.task_id,
                    issue_path=str(task.issue_path()),
                    verify_test_log=verify_log if verify_log.is_file() else None,
                    tests_passed=tests_passed,
                    agent_mode=task.agent.mode,
                    failure_mode=task.eval.failure_mode,
                    difficulty=task.eval.difficulty,
                    bug_count=task.eval.bug_count,
                    eval_tags=task.eval.tags or None,
                    workspace_patch=workspace_patch,
                ),
            )

    finished_at = datetime.now(timezone.utc)
    result = RunResult(
        task_id=task.task_id,
        base_commit=task.repo.base_commit,
        agent_mode=task.agent.mode,
        repo_path=repo_path,
        output_dir=output_dir,
        trajectory_path=trajectory_path,
        mini_exit_code=mini_exit_code,
        test_exit_code=test_exit_code,
        tests_passed=tests_passed,
        mini_command=mini_command,
    )
    write_run_meta(result, started_at=started_at, finished_at=finished_at)
    return result


def _get_typer_echo():
    try:
        import typer

        return typer.echo
    except ImportError:
        return print

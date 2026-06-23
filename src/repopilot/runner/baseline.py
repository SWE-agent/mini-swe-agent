"""Run upstream mini-swe-agent as baseline (Phase 1.5)."""

from __future__ import annotations

from pathlib import Path

import typer

from repopilot.runner.run_task import find_project_root, resolve_task_dir, run_benchmark_task

app = typer.Typer(help="Run upstream mini-swe-agent baseline on a benchmark task.")


@app.command("run")
def run(
    task: Path = typer.Argument(..., help="Task directory or task_id under benchmarks/"),
    project_root: Path | None = typer.Option(
        None,
        "--project-root",
        help="Repository root (default: git toplevel from cwd)",
    ),
    dry_run: bool = typer.Option(False, "--dry-run", help="Print planned commands without running"),
    skip_mini: bool = typer.Option(
        False,
        "--skip-mini",
        help="Prepare workspace and verify only; skip mini agent",
    ),
    no_restore: bool = typer.Option(
        False,
        "--no-restore",
        help="Keep the task git worktree after run (default: remove runs/{task_id}/.workspace)",
    ),
) -> None:
    """Run a benchmark task in baseline (mini) mode."""
    root = (project_root or find_project_root()).resolve()
    task_dir = resolve_task_dir(task, benchmarks_root=root / "benchmarks")
    result = run_benchmark_task(
        task_dir,
        project_root=root,
        skip_mini=skip_mini,
        restore_workspace=not no_restore,
        dry_run=dry_run,
        force_baseline=True,
    )
    if dry_run:
        return
    typer.echo(f"Baseline run for {result.task_id} finished.")
    typer.echo(f"  mini exit code: {result.mini_exit_code}")
    typer.echo(f"  verify exit code: {result.test_exit_code} ({'passed' if result.tests_passed else 'failed'})")
    typer.echo(f"  artifacts: {result.output_dir}/")


if __name__ == "__main__":
    app()

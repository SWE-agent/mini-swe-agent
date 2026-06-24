"""Benchmark task configuration schema."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field, model_validator


class RepoConfig(BaseModel):
    path: str | None = None
    repo_url: str | None = None
    base_commit: str
    setup_patch: str | None = None
    """Optional patch file (relative to task dir) applied after checkout to plant a known bug."""

    @model_validator(mode="after")
    def _path_xor_url(self) -> RepoConfig:
        if bool(self.path) == bool(self.repo_url):
            raise ValueError("Exactly one of repo.path or repo.repo_url must be set")
        return self


class EvalTags(BaseModel):
    failure_mode: str | None = None
    difficulty: str | None = None
    expected_repair_rounds: int | None = None
    tags: list[str] = Field(default_factory=list)


class AgentConfig(BaseModel):
    mode: Literal["baseline", "repopilot"] = "baseline"
    mini_flags: list[str] = Field(default_factory=lambda: ["-y", "--exit-immediately"])
    output_trajectory: str = "runs/{task_id}/trajectory.traj.json"

    def resolve_output_trajectory(self, task_id: str) -> Path:
        return Path(self.output_trajectory.format(task_id=task_id))


class TaskConfig(BaseModel):
    task_id: str
    description: str = ""
    repo: RepoConfig
    issue_file: str = "issue.md"
    test_command: str
    expected_behavior: str = ""
    eval: EvalTags = Field(default_factory=EvalTags)
    agent: AgentConfig = Field(default_factory=AgentConfig)

    @property
    def task_dir(self) -> Path | None:
        return getattr(self, "_task_dir", None)

    def issue_path(self) -> Path:
        if self.task_dir is None:
            raise ValueError("task_dir not set; load via load_task()")
        return self.task_dir / self.issue_file

    def setup_patch_path(self) -> Path | None:
        if self.repo.setup_patch is None:
            return None
        if self.task_dir is None:
            raise ValueError("task_dir not set; load via load_task()")
        return self.task_dir / self.repo.setup_patch

    def read_issue(self) -> str:
        return self.issue_path().read_text()


def load_task_config(data: dict, *, task_dir: Path | None = None) -> TaskConfig:
    task = TaskConfig.model_validate(data)
    if task_dir is not None:
        task_dir = task_dir.resolve()
        object.__setattr__(task, "_task_dir", task_dir)
        if task.task_id != task_dir.name:
            raise ValueError(f"task_id '{task.task_id}' must match directory name '{task_dir.name}'")
    return task


def load_task(task_dir: Path) -> TaskConfig:
    """Load and validate config.yaml from a benchmark task directory."""
    task_dir = task_dir.resolve()
    config_path = task_dir / "config.yaml"
    if not config_path.is_file():
        raise FileNotFoundError(f"Missing config.yaml in {task_dir}")
    data = yaml.safe_load(config_path.read_text())
    if not isinstance(data, dict):
        raise ValueError(f"config.yaml must be a mapping: {config_path}")
    task = load_task_config(data, task_dir=task_dir)
    if not task.issue_path().is_file():
        raise FileNotFoundError(f"Missing issue file: {task.issue_path()}")
    patch_path = task.setup_patch_path()
    if patch_path is not None and not patch_path.is_file():
        raise FileNotFoundError(f"Missing setup patch: {patch_path}")
    return task


def discover_tasks(benchmarks_root: Path) -> list[Path]:
    """Return task directories containing config.yaml, sorted by name."""
    benchmarks_root = benchmarks_root.resolve()
    if not benchmarks_root.is_dir():
        return []
    return sorted(
        path.parent
        for path in benchmarks_root.glob("*/config.yaml")
        if path.parent.is_dir()
    )

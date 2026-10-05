import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

from minisweagent import package_dir
from minisweagent.run.benchmarks.swebench import DATASET_MAPPING

INSTANCE_ID = "example__repo.abc.combine_file__task"


def _git(repo: Path, *args: str) -> str:
    return subprocess.run(["git", "-C", str(repo), *args], check=True, capture_output=True, text=True).stdout.strip()


@pytest.fixture
def task_repo(tmp_path: Path) -> Path:
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init", "-b", "main")
    _git(repo, "config", "user.name", "Test")
    _git(repo, "config", "user.email", "test@example.com")
    (repo / "source.py").write_text("original\n")
    _git(repo, "add", "source.py")
    _git(repo, "commit", "-m", "Original source")
    _git(repo, "checkout", "-b", INSTANCE_ID)
    (repo / "source.py").write_text("task bug\n")
    _git(repo, "commit", "-am", "Task bug")
    _git(repo, "checkout", "main")
    return repo


def _run_benchmark(
    tmp_path: Path,
    repo: Path,
    runner: str,
    subset: str,
    *,
    instance_ids: tuple[str, ...] = (INSTANCE_ID,),
    startup_command: str | None = None,
) -> tuple[str, str]:
    dataset = tmp_path / DATASET_MAPPING.get(subset, subset)
    dataset.mkdir(parents=True)
    (dataset / "train.json").write_text(
        json.dumps(
            [{"instance_id": instance_id, "problem_statement": "Fix the task bug"} for instance_id in instance_ids]
        )
    )
    config = {
        "environment": {"environment_class": "local", "cwd": str(repo)},
        "model": {"model_class": "deterministic", "model_name": "deterministic", "outputs": []},
    }
    if startup_command is not None:
        config["run"] = {"env_startup_command": startup_command}
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(config))
    output = tmp_path / "output"
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            f"minisweagent.run.benchmarks.{runner}",
            "--subset",
            subset,
            "--split",
            "train",
            "-c",
            str(config_path),
            "-o",
            str(output),
        ],
        cwd=tmp_path,
        env=os.environ
        | {
            "PYTHONPATH": str(package_dir.parent),
            "MSWEA_GLOBAL_CONFIG_DIR": str(tmp_path / "global_config"),
            "MSWEA_SILENT_STARTUP": "1",
            "HF_HOME": str(tmp_path / "hf_cache"),
            "HF_HUB_OFFLINE": "1",
            "HF_DATASETS_OFFLINE": "1",
            "LITELLM_LOCAL_MODEL_COST_MAP": "True",
        },
        capture_output=True,
        text=True,
        timeout=30,
    )
    logs = result.stdout + result.stderr
    if runner == "swebench":
        assert result.returncode == 0, logs
        statuses = yaml.safe_load(next(output.glob("exit_statuses_*.yaml")).read_text())
        assert list(statuses["instances_by_exit_status"].values()) == [list(instance_ids)], logs
        status = next(iter(statuses["instances_by_exit_status"]))
    else:
        assert result.returncode != 0, logs
        status = "RuntimeError" if "Error executing startup command" in logs else "ValidationError"
    if status == "ValidationError":
        assert "ValidationError" in logs and "system_template" in logs and "instance_template" in logs
    assert config_path.read_text() == yaml.safe_dump(config)
    return status, logs


@pytest.mark.parametrize(
    ("runner", "subset"),
    [(runner, subset) for runner in ("swebench", "swebench_single") for subset in ("smith", DATASET_MAPPING["smith"])],
)
def test_smith_checks_out_task_before_agent_initialization(tmp_path, task_repo, runner, subset):
    status, logs = _run_benchmark(tmp_path, task_repo, runner, subset)
    # Required agent templates are deliberately absent, so initialization stops before any model query.
    assert status == "ValidationError", logs
    assert "system_template" in logs and "instance_template" in logs
    assert _git(task_repo, "rev-parse", "HEAD") == _git(task_repo, "rev-parse", INSTANCE_ID)
    assert (task_repo / "source.py").read_text() == "task bug\n"


@pytest.mark.parametrize(
    ("runner", "subset"),
    [(runner, subset) for runner in ("swebench", "swebench_single") for subset in ("smith", DATASET_MAPPING["smith"])],
)
def test_smith_missing_task_ref_fails_before_agent_initialization(tmp_path, task_repo, runner, subset):
    status, logs = _run_benchmark(tmp_path, task_repo, runner, subset, instance_ids=("missing_task_ref",))
    assert status == "RuntimeError", logs
    assert "Error executing startup command" in logs and "missing_task_ref" in logs
    assert _git(task_repo, "branch", "--show-current") == "main"
    assert (task_repo / "source.py").read_text() == "original\n"


@pytest.mark.parametrize(
    ("runner", "subset", "startup_command"),
    [
        (runner, subset, startup_command)
        for runner in ("swebench", "swebench_single")
        for subset in ("smith", DATASET_MAPPING["smith"])
        for startup_command in ("", "git checkout main && printf configured > startup.txt")
    ],
)
def test_smith_explicit_startup_command_overrides_default(tmp_path, task_repo, runner, subset, startup_command):
    status, logs = _run_benchmark(tmp_path, task_repo, runner, subset, startup_command=startup_command)
    assert status == "ValidationError", logs
    assert _git(task_repo, "branch", "--show-current") == "main"
    if startup_command:
        assert (task_repo / "startup.txt").read_text() == "configured"
    else:
        assert not (task_repo / "startup.txt").exists()


@pytest.mark.parametrize(
    ("runner", "subset"),
    [
        (runner, subset)
        for runner in ("swebench", "swebench_single")
        for subset in ("lite", "rebench", "custom_dataset")
    ],
)
def test_other_datasets_do_not_get_smith_startup(tmp_path, task_repo, runner, subset):
    status, logs = _run_benchmark(tmp_path, task_repo, runner, subset)
    assert status == "ValidationError", logs
    assert _git(task_repo, "branch", "--show-current") == "main"
    assert (task_repo / "source.py").read_text() == "original\n"


def test_smith_batch_renders_startup_separately_for_each_instance(tmp_path, task_repo):
    second_instance = INSTANCE_ID + "_second"
    _git(task_repo, "checkout", "-b", second_instance)
    (task_repo / "source.py").write_text("second task bug\n")
    _git(task_repo, "commit", "-am", "Second task bug")
    _git(task_repo, "checkout", "main")
    status, logs = _run_benchmark(tmp_path, task_repo, "swebench", "smith", instance_ids=(INSTANCE_ID, second_instance))
    assert status == "ValidationError", logs
    assert (task_repo / "source.py").read_text() == "second task bug\n"
    assert _git(task_repo, "reflog", "-2", "--format=%gs").splitlines() == [
        f"checkout: moving from {INSTANCE_ID} to {second_instance}",
        f"checkout: moving from main to {INSTANCE_ID}",
    ]

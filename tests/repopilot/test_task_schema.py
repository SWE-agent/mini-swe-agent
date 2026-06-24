from pathlib import Path

import pytest
from pydantic import ValidationError

from repopilot.schema import TaskConfig, discover_tasks, load_task, load_task_config

BENCHMARKS = Path(__file__).resolve().parents[2] / "benchmarks"
TASK_001 = BENCHMARKS / "task_001_sudoku"


def test_load_task_001_sudoku():
    task = load_task(TASK_001)
    assert task.task_id == "task_001_sudoku"
    assert task.repo.path == "."
    assert task.repo.base_commit == "179e790"
    assert task.repo.setup_patch == "setup.patch"
    assert task.agent.mode == "baseline"
    assert "upstream/tests/run/test_sudoku.py" in task.test_command
    assert "upstream/src/minisweagent/run/sudoku.py" in task.read_issue()
    assert task.setup_patch_path().is_file()
    assert task.agent.resolve_output_trajectory(task.task_id) == Path("runs/task_001_sudoku/trajectory.traj.json")
    assert task.eval.failure_mode == "off_by_one"
    assert task.eval.difficulty == "single_file"


def test_task_id_must_match_directory_name():
    data = load_task(TASK_001).model_dump()
    data["task_id"] = "wrong_name"
    (TASK_001 / "config.yaml").read_text()  # ensure task exists
    with pytest.raises(ValueError, match="must match directory name"):
        load_task_config(data, task_dir=TASK_001)


def test_repo_requires_path_or_url_not_both():
    with pytest.raises(ValidationError, match="Exactly one"):
        TaskConfig.model_validate(
            {
                "task_id": "t",
                "repo": {"path": ".", "repo_url": "https://example.com", "base_commit": "abc"},
                "test_command": "pytest",
            }
        )


def test_discover_tasks_includes_task_001():
    tasks = discover_tasks(BENCHMARKS)
    assert TASK_001 in tasks


@pytest.mark.parametrize(
    "task_id,failure_mode",
    [
        ("task_006_sudoku_logic", "logic"),
        ("task_007_eval_import", "import_path"),
        ("task_008_expr_divzero", "wrong_condition"),
        ("task_009_serialize_none", "null_handling"),
        ("task_010_expr_whitespace", "logic"),
    ],
)
def test_load_new_benchmark_tasks(task_id, failure_mode):
    task_dir = BENCHMARKS / task_id
    task = load_task(task_dir)
    assert task.task_id == task_id
    assert task.eval.failure_mode == failure_mode
    assert task.setup_patch_path().is_file()


def test_discover_tasks_count():
    tasks = discover_tasks(BENCHMARKS)
    assert len(tasks) == 14


@pytest.mark.parametrize(
    "task_id,bug_count",
    [
        ("task_011_sudoku_multi3", 3),
        ("task_012_expr_multi4", 4),
        ("task_013_sudoku_multi5", 5),
        ("task_014_expr_multi5", 5),
    ],
)
def test_load_multi_bug_tasks(task_id, bug_count):
    task_dir = BENCHMARKS / task_id
    task = load_task(task_dir)
    assert task.task_id == task_id
    assert task.eval.failure_mode == "composite"
    assert task.eval.bug_count == bug_count
    assert "multi_bug" in task.eval.tags
    assert task.setup_patch_path().is_file()

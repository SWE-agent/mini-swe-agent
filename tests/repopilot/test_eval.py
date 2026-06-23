import json
from pathlib import Path

import pytest
import yaml

from repopilot.eval import (
    aggregate_runs,
    discover_run_dirs,
    failure_breakdown,
    load_run_record,
    write_eval_summary,
)
from repopilot.eval.loader import RunRecord, classify_run

ROOT = Path(__file__).resolve().parents[2]
RUNS = ROOT / "runs"
TASK_RUN = RUNS / "task_001_sudoku"


def _minimal_trace(task_id: str = "task_test", *, post_fix_rc: int = 0) -> dict:
    return {
        "schema_version": "1.0",
        "task_id": task_id,
        "agent_mode": "baseline",
        "model": "test-model",
        "exit_status": "Submitted",
        "metrics": {"api_calls": 2, "instance_cost": 0.01, "step_count": 2, "repair_rounds": 1},
        "steps": [
            {
                "step": 1,
                "reasoning": "run tests",
                "tool_calls": [{"command": "pytest -v", "returncode": 1}],
            },
            {
                "step": 2,
                "reasoning": "fix",
                "tool_calls": [
                    {"command": "path.write_text(...)", "returncode": 0},
                    {"command": "pytest -v", "returncode": post_fix_rc},
                ],
            },
        ],
        "pytest_runs": [
            {"phase": "pre_fix", "returncode": 1, "summary": "1 failed", "log": "1 failed, 2 passed"},
            {"phase": "post_fix", "returncode": post_fix_rc, "summary": "3 passed", "log": "3 passed"},
        ],
        "patch": {"source": "test", "text": "diff --git a/foo b/foo\n"},
    }


@pytest.fixture
def sample_run(tmp_path):
    run_dir = tmp_path / "task_sample"
    run_dir.mkdir()
    (run_dir / "trace.json").write_text(json.dumps(_minimal_trace()))
    (run_dir / "run_meta.yaml").write_text(
        yaml.safe_dump(
            {
                "task_id": "task_sample",
                "agent_mode": "baseline",
                "tests_passed": True,
                "test_exit_code": 0,
                "base_commit": "abc123",
            }
        )
    )
    return run_dir


def test_discover_run_dirs_excludes_eval(tmp_path, sample_run):
    (tmp_path / "eval").mkdir()
    (tmp_path / "eval" / "trace.json").write_text("{}")
    found = discover_run_dirs(tmp_path)
    assert sample_run in found
    assert tmp_path / "eval" not in found


def test_load_run_record_success(sample_run):
    record = load_run_record(sample_run)
    assert record.task_id == "task_sample"
    assert record.outcome == "success"
    assert record.failure_category is None


def test_classify_verify_mismatch():
    record = RunRecord(
        task_id="t",
        agent_mode="baseline",
        model="m",
        run_dir=Path("."),
        tests_passed=False,
        pytest_runs=[
            {"phase": "pre_fix", "returncode": 1, "log": ""},
            {"phase": "post_fix", "returncode": 0, "log": "3 passed"},
        ],
        steps=[
            {"step": 1, "tool_calls": [{"command": "path.write_text(x)", "returncode": 0}]},
            {"step": 2, "tool_calls": [{"command": "pytest", "returncode": 0}]},
        ],
        has_patch=True,
    )
    classified = classify_run(record)
    assert classified.outcome == "failure"
    assert classified.failure_category == "verify_mismatch"


def test_aggregate_and_breakdown(sample_run):
    record = load_run_record(sample_run)
    metrics = aggregate_runs([record])
    assert metrics["total_runs"] == 1
    assert metrics["success_count"] == 1
    breakdown = failure_breakdown([record])
    assert breakdown["failed_runs"] == 0


def test_write_eval_summary(tmp_path, sample_run):
    out = write_eval_summary(tmp_path, output_dir=tmp_path / "eval" / "summary")
    assert (out / "eval_report.md").is_file()
    assert (out / "metrics.json").is_file()
    assert (out / "failure_breakdown.md").is_file()
    assert (tmp_path / "eval" / "task_sample" / "run_summary.md").is_file()


@pytest.mark.skipif(not TASK_RUN.is_dir(), reason="requires local task_001 run")
def test_load_real_task_001():
    record = load_run_record(TASK_RUN)
    assert record.task_id == "task_001_sudoku"
    assert record.step_count >= 1

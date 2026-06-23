from pathlib import Path

import pytest

from repopilot.trace import (
    extract_patch_diff,
    extract_pytest_runs,
    iter_tool_rows,
    load_trajectory,
    record_trace,
    TraceContext,
)

ROOT = Path(__file__).resolve().parents[2]
TRAJ = ROOT / "runs" / "task_001_sudoku" / "trajectory.traj.json"


pytestmark = pytest.mark.skipif(not TRAJ.is_file(), reason="requires local task_001 trajectory")


def test_load_trajectory_has_messages():
    traj = load_trajectory(TRAJ)
    assert "messages" in traj
    assert "info" in traj
    assert len(traj["messages"]) > 0


def test_extract_pytest_runs_from_task_001():
    rows = iter_tool_rows(load_trajectory(TRAJ)["messages"])
    runs = extract_pytest_runs(rows)
    assert len(runs) >= 2
    assert runs[0].phase == "pre_fix"
    assert runs[0].returncode == 1
    assert runs[-1].phase == "post_fix"
    assert runs[-1].returncode == 0


def test_extract_patch_from_task_001():
    rows = iter_tool_rows(load_trajectory(TRAJ)["messages"])
    patch, source = extract_patch_diff(rows)
    assert source in {"reconstructed from edit command", "git diff in trajectory"}
    assert "value" in patch


def test_record_trace_writes_artifacts(tmp_path):
    artifacts = record_trace(
        TRAJ,
        tmp_path,
        ctx=TraceContext(task_id="task_001_sudoku", tests_passed=True),
    )
    assert artifacts.trace_json.is_file()
    assert artifacts.patch_diff.is_file()
    assert artifacts.test_log.is_file()
    assert artifacts.final_report.is_file()

    trace = artifacts.trace_json.read_text()
    assert "task_001_sudoku" in trace
    assert "api_calls" in trace
    assert "steps" in trace

    report = artifacts.final_report.read_text()
    assert "# RepoPilot Run Report" in report
    assert "Step 1" in report

    test_log = artifacts.test_log.read_text()
    assert "pre_fix" in test_log
    assert "post_fix" in test_log

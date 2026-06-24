import json
from pathlib import Path

import pytest
import yaml

from repopilot.eval import (
    aggregate_runs,
    comparison_rows,
    discover_run_dirs,
    extract_failure_snippet,
    failure_breakdown,
    load_run_record,
    load_task_runs,
    tag_breakdown,
    trajectory_metrics,
    write_eval_summary,
)
from repopilot.eval.compare import render_task_run_comparison
from repopilot.eval.failure_analysis import render_tag_breakdown
from repopilot.eval.loader import RunRecord, classify_run
from repopilot.trace.classify import classify_trace, detect_wrong_file_edited

ROOT = Path(__file__).resolve().parents[2]
RUNS = ROOT / "runs"
TASK_RUN = RUNS / "task_001_sudoku"

SAMPLE_PYTEST_LOG = """\
============================= test session starts ==============================
collected 5 items

upstream/tests/run/test_eval_module.py::test_eval_basic_arithmetic FAILED [ 20%]

=================================== FAILURES ===================================
__________________________ test_eval_basic_arithmetic __________________________

    def test_eval_basic_arithmetic():
>       assert eval_expr("4*5") == 20
E       AssertionError: assert 9.0 == 20

upstream/tests/run/test_eval_module.py:9: AssertionError
========================= 1 failed, 4 passed in 0.02s ==========================
"""


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
    assert (out / "comparison_table.csv").is_file()
    assert (tmp_path / "eval" / "compare" / "comparison_report.md").is_file()
    assert (tmp_path / "eval" / "task_sample" / "run_summary.md").is_file()
    assert (tmp_path / "eval" / "task_sample" / "trajectory_analysis.md").is_file()


def test_trajectory_metrics(sample_run):
    record = load_run_record(sample_run)
    metrics = trajectory_metrics(record)
    assert metrics["step_count"] == 2
    assert metrics["steps_to_first_edit"] == 2
    assert metrics["first_test_step"] == 1
    assert metrics["tests_before_edit"] == 1


def test_comparison_rows(sample_run):
    record = load_run_record(sample_run)
    rows = comparison_rows([record])
    assert len(rows) == 1
    assert rows[0]["task_id"] == "task_sample"
    assert rows[0]["steps_to_first_edit"] == 2


def test_tag_breakdown(sample_run):
    record = load_run_record(sample_run)
    record.failure_mode = "wrong_operator"
    record.difficulty = "single_file"
    breakdown = tag_breakdown([record])
    assert breakdown["by_failure_mode"]["wrong_operator"] == 1
    assert breakdown["by_difficulty"]["single_file"] == 1


@pytest.mark.skipif(not TASK_RUN.is_dir(), reason="requires local task_001 run")
def test_load_real_task_001():
    record = load_run_record(TASK_RUN)
    assert record.task_id == "task_001_sudoku"
    assert record.step_count >= 1


def test_load_task_runs_with_history(tmp_path):
    task_dir = tmp_path / "task_demo"
    task_dir.mkdir()
    _write_run(task_dir, agent_mode="baseline", cost=0.01, steps=2, label_meta="latest")

    history = task_dir / "history" / "skip_mini"
    history.mkdir(parents=True)
    _write_run(history, agent_mode="baseline", cost=0.001, steps=0, label_meta="skip_mini")

    records = load_task_runs(tmp_path, "task_demo")
    assert len(records) == 2
    assert records[0].run_label == "latest"
    assert records[1].run_label == "skip_mini"
    report = render_task_run_comparison(records, "task_demo")
    assert "Multi-run deltas" in report
    assert "skip_mini vs latest" in report


def test_compare_task_run_delta_cost(tmp_path):
    task_dir = tmp_path / "task_demo"
    task_dir.mkdir()
    _write_run(task_dir, agent_mode="baseline", cost=0.05, steps=4)
    archived = task_dir / "history" / "repair_v1"
    archived.mkdir(parents=True)
    _write_run(archived, agent_mode="repopilot", cost=0.08, steps=6)

    report = render_task_run_comparison(load_task_runs(tmp_path, "task_demo"), "task_demo")
    assert "+0.03" in report or "+0.0300" in report
    assert "repair_v1 vs latest" in report


def test_breakdown_by_failure_mode(tmp_path, sample_run):
    record = load_run_record(sample_run)
    record.failure_mode = "wrong_operator"
    md = render_tag_breakdown(tag_breakdown([record]), by="failure_mode")
    assert "by failure_mode" in md
    assert "`wrong_operator`" in md
    assert "By difficulty" not in md


def test_exemplar_tests_still_failing(tmp_path):
    """Correct file edited but fix wrong → tests_still_failing exemplar from test.log."""
    run_dir = tmp_path / "task_fail"
    run_dir.mkdir()
    trace = _minimal_trace("task_fail", post_fix_rc=1)
    trace["pytest_runs"] = [
        {
            "phase": "pre_fix",
            "returncode": 1,
            "summary": "1 failed",
            "log": SAMPLE_PYTEST_LOG,
            "failed_tests": [
                {
                    "test": "test_eval_basic_arithmetic",
                    "assertion": "AssertionError: assert 9.0 == 20",
                    "file_line": "upstream/tests/run/test_eval_module.py:9",
                }
            ],
        },
        {"phase": "post_fix", "returncode": 1, "summary": "1 failed", "log": SAMPLE_PYTEST_LOG},
    ]
    trace["steps"][1]["tool_calls"][0]["command"] = (
        "p = Path('upstream/src/minisweagent/run/eval_module.py'); p.write_text('...')"
    )
    trace["steps"][1]["files_touched"] = ["upstream/src/minisweagent/run/eval_module.py"]
    (run_dir / "trace.json").write_text(json.dumps(trace))
    (run_dir / "run_meta.yaml").write_text(yaml.safe_dump({"task_id": "task_fail", "tests_passed": False}))
    (run_dir / "test.log").write_text(SAMPLE_PYTEST_LOG)

    record = load_run_record(run_dir)
    assert record.failure_category == "tests_still_failing"
    snippet = extract_failure_snippet(record)
    assert "test_eval_basic_arithmetic" in snippet
    assert "AssertionError" in snippet

    breakdown = tag_breakdown([record])
    md = render_tag_breakdown(breakdown)
    assert "Exemplars — `tests_still_failing`" in md


def test_exemplar_wrong_file_edited(tmp_path):
    """Edit unrelated file while test points elsewhere → wrong_file_edited exemplar."""
    run_dir = tmp_path / "task_wrong_file"
    run_dir.mkdir()
    trace = _minimal_trace("task_wrong_file", post_fix_rc=1)
    trace["pytest_runs"] = [
        {
            "phase": "pre_fix",
            "returncode": 1,
            "summary": "1 failed",
            "log": SAMPLE_PYTEST_LOG,
            "failed_tests": [
                {
                    "test": "test_eval_basic_arithmetic",
                    "file_line": "upstream/tests/run/test_eval_module.py:9",
                }
            ],
        },
        {"phase": "post_fix", "returncode": 1, "summary": "1 failed", "log": SAMPLE_PYTEST_LOG},
    ]
    trace["steps"][1]["tool_calls"][0]["command"] = (
        "p = Path('upstream/src/minisweagent/run/expr/tokenize.py'); p.write_text('...')"
    )
    trace["steps"][1]["files_touched"] = ["upstream/src/minisweagent/run/expr/tokenize.py"]
    (run_dir / "trace.json").write_text(json.dumps(trace))
    (run_dir / "run_meta.yaml").write_text(yaml.safe_dump({"task_id": "task_wrong_file", "tests_passed": False}))
    (run_dir / "test.log").write_text(SAMPLE_PYTEST_LOG)

    record = load_run_record(run_dir)
    assert detect_wrong_file_edited(steps=record.steps, pytest_runs=record.pytest_runs)
    assert record.failure_category == "wrong_file_edited"
    snippet = extract_failure_snippet(record)
    assert "tokenize.py" in snippet
    assert "test_eval_module.py" in snippet

    md = render_tag_breakdown(tag_breakdown([record]), by="failure_category")
    assert "Exemplars — `wrong_file_edited`" in md


def _task_id_for_run_dir(run_dir: Path) -> str:
    if run_dir.parent.name == "history":
        return run_dir.parent.parent.name
    return run_dir.name


def _write_run(
    run_dir: Path,
    *,
    agent_mode: str,
    cost: float,
    steps: int,
    label_meta: str | None = None,
) -> None:
    trace = _minimal_trace(_task_id_for_run_dir(run_dir))
    trace["agent_mode"] = agent_mode
    trace["metrics"]["instance_cost"] = cost
    trace["metrics"]["step_count"] = steps
    (run_dir / "trace.json").write_text(json.dumps(trace))
    meta = {
        "task_id": trace["task_id"],
        "agent_mode": agent_mode,
        "tests_passed": True,
        "test_exit_code": 0,
    }
    if label_meta:
        meta["run_label"] = label_meta
    (run_dir / "run_meta.yaml").write_text(yaml.safe_dump(meta))

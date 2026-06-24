"""Tests for Phase 5 visualization."""

import json
from pathlib import Path

import yaml

from repopilot.eval.loader import load_run_record
from repopilot.eval.trajectory_analysis import render_trajectory_analysis
from repopilot.eval.visualize import (
    render_ascii_bar_chart,
    render_failure_distribution_charts,
    render_mermaid_source,
    render_mermaid_timeline,
    render_run_html,
    write_run_view,
)


def _minimal_trace(task_id: str = "task_viz") -> dict:
    return {
        "schema_version": "2.0",
        "task_id": task_id,
        "agent_mode": "baseline",
        "model": "test-model",
        "exit_status": "Submitted",
        "outcome": "failure",
        "failure_category": "tests_still_failing",
        "failed_step": 2,
        "metrics": {"api_calls": 2, "instance_cost": 0.02, "step_count": 3, "repair_rounds": 1},
        "steps": [
            {"step": 1, "stage": "test", "reasoning": "run tests", "tool_calls": [{"command": "pytest", "returncode": 1}]},
            {
                "step": 2,
                "stage": "edit",
                "reasoning": "apply fix",
                "files_touched": ["upstream/src/foo.py"],
                "tool_calls": [{"command": "p.write_text(...)", "returncode": 0}],
            },
            {"step": 3, "stage": "test", "reasoning": "verify", "tool_calls": [{"command": "pytest", "returncode": 1}]},
        ],
        "pytest_runs": [
            {"phase": "pre_fix", "returncode": 1, "summary": "1 failed", "log": "1 failed"},
            {"phase": "post_fix", "returncode": 1, "summary": "1 failed", "log": "1 failed"},
        ],
        "patch": {"source": "test", "text": "diff --git a/foo b/foo\n"},
    }


def _sample_run(tmp_path: Path) -> Path:
    run_dir = tmp_path / "task_viz"
    run_dir.mkdir()
    (run_dir / "trace.json").write_text(json.dumps(_minimal_trace()))
    (run_dir / "run_meta.yaml").write_text(
        yaml.safe_dump({"task_id": "task_viz", "tests_passed": False, "test_exit_code": 1})
    )
    (run_dir / "patch.diff").write_text("diff --git a/foo b/foo\n+fixed\n")
    return run_dir


def test_render_mermaid_timeline(tmp_path):
    record = load_run_record(_sample_run(tmp_path))
    md = render_mermaid_timeline(record)
    assert "```mermaid" in md
    assert "flowchart LR" in md
    assert "s1" in md and "s2" in md
    assert "class s2 failed" in md

    analysis = render_trajectory_analysis(record)
    assert "## Trajectory timeline (Mermaid)" in analysis


def test_render_ascii_bar_chart():
    chart = render_ascii_bar_chart({"tests_still_failing": 3, "verify_mismatch": 1})
    assert "tests_still_failing" in chart
    assert "█" in chart


def test_render_failure_distribution_charts():
    md = render_failure_distribution_charts(
        {"by_category": {"tests_still_failing": 2}, "by_stage": {"test": 2}, "failed_runs": 2}
    )
    assert "## Failure distribution" in md
    assert "tests_still_failing" in md


def test_render_run_html_and_write(tmp_path):
    run_dir = _sample_run(tmp_path)
    record = load_run_record(run_dir)
    html = render_run_html(record)
    assert "<!DOCTYPE html>" in html
    assert "task_viz" in html
    assert "flowchart LR" in html
    assert "apply fix" in html
    assert "diff --git" in html

    out = write_run_view(record, tmp_path / "out" / "view.html")
    assert out.is_file()
    assert out.read_text().startswith("<!DOCTYPE html>")

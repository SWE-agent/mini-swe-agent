"""Load and normalize benchmark run records for evaluation."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import yaml


@dataclass
class RunRecord:
    task_id: str
    agent_mode: str
    model: str
    run_dir: Path
    trace_path: Path | None = None
    meta_path: Path | None = None

    api_calls: int = 0
    instance_cost: float = 0.0
    step_count: int = 0
    repair_rounds: int = 0
    exit_status: str = ""

    tests_passed: bool | None = None
    test_exit_code: int | None = None
    base_commit: str | None = None
    started_at: str | None = None
    finished_at: str | None = None

    outcome: str = "unknown"
    failure_category: str | None = None
    failure_stage: str | None = None
    failed_step: int | None = None
    failure_message: str | None = None

    pytest_runs: list[dict] = field(default_factory=list)
    steps: list[dict] = field(default_factory=list)
    patch_source: str | None = None
    has_patch: bool = False


def discover_run_dirs(runs_dir: Path) -> list[Path]:
    """Return task run directories containing trace.json, excluding eval output."""
    runs_dir = runs_dir.resolve()
    if not runs_dir.is_dir():
        return []
    results: list[Path] = []
    for path in sorted(runs_dir.iterdir()):
        if not path.is_dir():
            continue
        if path.name == "eval":
            continue
        if (path / "trace.json").is_file():
            results.append(path)
    return results


def _load_yaml(path: Path) -> dict:
    data = yaml.safe_load(path.read_text())
    return data if isinstance(data, dict) else {}


def _has_edit_command(steps: list[dict]) -> bool:
    for step in steps:
        for tc in step.get("tool_calls", []):
            cmd = tc.get("command", "")
            if any(token in cmd for token in ("path.write_text", "sed -i", "git apply", "patch")):
                return True
    return False


def _agent_post_fix_passed(pytest_runs: list[dict]) -> bool | None:
    post = next((r for r in pytest_runs if r.get("phase") == "post_fix"), None)
    if post is None:
        return None
    return post.get("returncode") == 0


def _last_pytest_summary(pytest_runs: list[dict]) -> str:
    if not pytest_runs:
        return ""
    last = pytest_runs[-1]
    summary = last.get("summary") or ""
    if summary:
        return summary
    log = last.get("log", "")
    for line in reversed(log.splitlines()):
        stripped = line.strip().strip("=")
        if "passed" in stripped or "failed" in stripped:
            return stripped
    return ""


def _find_failed_step(steps: list[dict], *, tests_passed: bool | None) -> int | None:
    if not steps:
        return None
    if tests_passed is False:
        for step in reversed(steps):
            for tc in step.get("tool_calls", []):
                cmd = tc.get("command", "")
                if "pytest" in cmd and tc.get("returncode") not in (None, 0):
                    return step.get("step")
        for step in reversed(steps):
            for tc in step.get("tool_calls", []):
                if tc.get("returncode") not in (None, 0):
                    return step.get("step")
    return steps[-1].get("step")


def classify_run(record: RunRecord) -> RunRecord:
    """Assign outcome, failure_category, failure_stage, failed_step."""
    agent_post_fix = _agent_post_fix_passed(record.pytest_runs)
    has_pytest = bool(record.pytest_runs)
    has_edit = _has_edit_command(record.steps)
    patch_text = record.has_patch

    if record.tests_passed is True:
        record.outcome = "success"
        record.failure_category = None
        record.failure_stage = None
        record.failed_step = None
        record.failure_message = None
        return record

    record.outcome = "failure"

    if agent_post_fix is True and record.tests_passed is False:
        record.failure_category = "verify_mismatch"
        record.failure_stage = "test"
        record.failure_message = "Agent post-fix pytest passed; runner verify failed"
    elif not has_pytest:
        record.failure_category = "tests_never_run"
        record.failure_stage = "read" if has_edit else "plan"
        record.failure_message = "No pytest run found in trajectory"
    elif not patch_text and not has_edit:
        record.failure_category = "patch_empty"
        record.failure_stage = "edit"
        record.failure_message = "No patch or edit command in trajectory"
    elif _agent_post_fix_passed(record.pytest_runs) is False or (
        record.pytest_runs and record.pytest_runs[-1].get("returncode") not in (None, 0)
    ):
        record.failure_category = "tests_still_failing"
        record.failure_stage = "test"
        record.failure_message = _last_pytest_summary(record.pytest_runs) or "Tests still failing"
    elif "cost" in record.exit_status.lower() or "limit" in record.exit_status.lower():
        record.failure_category = "cost_limit_exceeded"
        record.failure_stage = "submit"
        record.failure_message = record.exit_status
    elif not has_edit and record.exit_status == "Submitted":
        record.failure_category = "agent_exited_early"
        record.failure_stage = "edit"
        record.failure_message = "Submitted without editing source"
    else:
        record.failure_category = "unknown"
        record.failure_stage = "submit"
        record.failure_message = record.exit_status or "Unknown failure"

    record.failed_step = _find_failed_step(record.steps, tests_passed=record.tests_passed)
    return record


def load_run_record(run_dir: Path) -> RunRecord:
    run_dir = run_dir.resolve()
    trace_path = run_dir / "trace.json"
    meta_path = run_dir / "run_meta.yaml"

    trace = json.loads(trace_path.read_text())
    meta = _load_yaml(meta_path) if meta_path.is_file() else {}

    metrics = trace.get("metrics", {})
    patch = trace.get("patch", {})

    record = RunRecord(
        task_id=meta.get("task_id") or trace.get("task_id") or run_dir.name,
        agent_mode=trace.get("agent_mode") or meta.get("agent_mode", "unknown"),
        model=trace.get("model", "unknown"),
        run_dir=run_dir,
        trace_path=trace_path,
        meta_path=meta_path if meta_path.is_file() else None,
        api_calls=int(metrics.get("api_calls") or 0),
        instance_cost=float(metrics.get("instance_cost") or 0.0),
        step_count=int(metrics.get("step_count") or len(trace.get("steps", []))),
        repair_rounds=int(metrics.get("repair_rounds") or 0),
        exit_status=str(trace.get("exit_status") or ""),
        tests_passed=meta.get("tests_passed") if meta else metrics.get("tests_passed"),
        test_exit_code=meta.get("test_exit_code"),
        base_commit=meta.get("base_commit"),
        started_at=meta.get("started_at"),
        finished_at=meta.get("finished_at"),
        pytest_runs=list(trace.get("pytest_runs") or []),
        steps=list(trace.get("steps") or []),
        patch_source=patch.get("source"),
        has_patch=bool((patch.get("text") or "").strip() and not patch.get("text", "").startswith("#")),
    )
    return classify_run(record)


def load_all_runs(runs_dir: Path) -> list[RunRecord]:
    return [load_run_record(path) for path in discover_run_dirs(runs_dir)]

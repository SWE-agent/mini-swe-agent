"""Load and normalize benchmark run records for evaluation."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import yaml

from repopilot.trace.classify import classify_trace


@dataclass
class RunRecord:
    task_id: str
    agent_mode: str
    model: str
    run_dir: Path
    run_label: str = "latest"
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

    failure_mode: str | None = None
    difficulty: str | None = None
    eval_tags: list[str] = field(default_factory=list)

    pytest_runs: list[dict] = field(default_factory=list)
    steps: list[dict] = field(default_factory=list)
    retrieved_files: list[str] = field(default_factory=list)
    patch_source: str | None = None
    has_patch: bool = False


def discover_run_dirs(runs_dir: Path) -> list[Path]:
    """Return latest task run directories containing trace.json, excluding eval output."""
    return [path for path, _label in discover_all_run_paths(runs_dir) if _label == "latest"]


def discover_all_run_paths(runs_dir: Path) -> list[tuple[Path, str]]:
    """Return (run_dir, run_label) pairs for latest runs and archived history runs."""
    runs_dir = runs_dir.resolve()
    if not runs_dir.is_dir():
        return []
    results: list[tuple[Path, str]] = []
    for path in sorted(runs_dir.iterdir()):
        if not path.is_dir() or path.name == "eval":
            continue
        if (path / "trace.json").is_file():
            results.append((path, "latest"))
        history = path / "history"
        if history.is_dir():
            for archived in sorted(history.iterdir()):
                if archived.is_dir() and (archived / "trace.json").is_file():
                    results.append((archived, archived.name))
    return results


def resolve_task_run_dir(runs_dir: Path, task_id: str) -> Path | None:
    runs_dir = runs_dir.resolve()
    exact = runs_dir / task_id
    if exact.is_dir() and (exact / "trace.json").is_file():
        return exact
    matches = sorted(
        path
        for path in runs_dir.iterdir()
        if path.is_dir() and path.name != "eval" and task_id in path.name and (path / "trace.json").is_file()
    )
    return matches[0] if len(matches) == 1 else None


def _load_yaml(path: Path) -> dict:
    data = yaml.safe_load(path.read_text())
    return data if isinstance(data, dict) else {}


def _trace_schema_version(trace: dict) -> float:
    raw = trace.get("schema_version", "1.0")
    try:
        return float(str(raw))
    except ValueError:
        return 1.0


def classify_run(record: RunRecord) -> RunRecord:
    """Assign outcome fields; prefer values already stored in trace v2."""
    if record.tests_passed is True and record.outcome != "success":
        classified = classify_trace(
            steps=record.steps,
            pytest_runs=record.pytest_runs,
            patch={"text": "diff" if record.has_patch else "", "source": record.patch_source or ""},
            exit_status=record.exit_status,
            tests_passed=True,
        )
        record.outcome = classified["outcome"]
        record.failure_category = classified.get("failure_category")
        record.failure_stage = classified.get("failure_stage")
        record.failed_step = classified.get("failed_step")
        record.failure_message = classified.get("failure_message")
        return record

    if record.outcome != "unknown":
        if record.outcome == "success":
            record.failure_category = None
            record.failure_stage = None
            record.failed_step = None
            record.failure_message = None
            return record
        if record.failure_category:
            return record

    classified = classify_trace(
        steps=record.steps,
        pytest_runs=record.pytest_runs,
        patch={"text": "diff" if record.has_patch else "", "source": record.patch_source or ""},
        exit_status=record.exit_status,
        tests_passed=record.tests_passed,
    )
    record.outcome = classified["outcome"]
    record.failure_category = classified.get("failure_category")
    record.failure_stage = classified.get("failure_stage")
    record.failed_step = classified.get("failed_step")
    record.failure_message = classified.get("failure_message")
    return record


def load_run_record(run_dir: Path, *, run_label: str | None = None) -> RunRecord:
    run_dir = run_dir.resolve()
    trace_path = run_dir / "trace.json"
    meta_path = run_dir / "run_meta.yaml"

    trace = json.loads(trace_path.read_text())
    meta = _load_yaml(meta_path) if meta_path.is_file() else {}

    metrics = trace.get("metrics", {})
    patch = trace.get("patch", {})
    task_tags = trace.get("task_tags") or {}

    tests_passed = meta.get("tests_passed") if meta else metrics.get("tests_passed")
    schema_v = _trace_schema_version(trace)
    label = run_label or ("latest" if run_dir.parent.name != "history" else run_dir.name)

    record = RunRecord(
        task_id=meta.get("task_id") or trace.get("task_id") or _task_id_from_run_dir(run_dir),
        agent_mode=trace.get("agent_mode") or meta.get("agent_mode", "unknown"),
        model=trace.get("model", "unknown"),
        run_dir=run_dir,
        run_label=label,
        trace_path=trace_path,
        meta_path=meta_path if meta_path.is_file() else None,
        api_calls=int(metrics.get("api_calls") or 0),
        instance_cost=float(metrics.get("instance_cost") or 0.0),
        step_count=int(metrics.get("step_count") or len(trace.get("steps", []))),
        repair_rounds=int(metrics.get("repair_rounds") or 0),
        exit_status=str(trace.get("exit_status") or ""),
        tests_passed=tests_passed,
        test_exit_code=meta.get("test_exit_code"),
        base_commit=meta.get("base_commit"),
        started_at=meta.get("started_at"),
        finished_at=meta.get("finished_at"),
        outcome=str(trace.get("outcome") or "unknown") if schema_v >= 2.0 else "unknown",
        failure_category=trace.get("failure_category") if schema_v >= 2.0 else None,
        failure_stage=trace.get("failure_stage") if schema_v >= 2.0 else None,
        failed_step=trace.get("failed_step") if schema_v >= 2.0 else None,
        failure_message=trace.get("failure_message") if schema_v >= 2.0 else None,
        failure_mode=task_tags.get("failure_mode"),
        difficulty=task_tags.get("difficulty"),
        eval_tags=list(task_tags.get("tags") or []),
        pytest_runs=list(trace.get("pytest_runs") or []),
        steps=list(trace.get("steps") or []),
        retrieved_files=list(trace.get("retrieved_files") or []),
        patch_source=patch.get("source"),
        has_patch=bool((patch.get("text") or "").strip() and not patch.get("text", "").startswith("#")),
    )
    return classify_run(record)


def _task_id_from_run_dir(run_dir: Path) -> str:
    if run_dir.parent.name == "history":
        return run_dir.parent.parent.name
    return run_dir.name


def load_all_runs(runs_dir: Path) -> list[RunRecord]:
    return [load_run_record(path, run_label=label) for path, label in discover_all_run_paths(runs_dir) if label == "latest"]


def load_task_runs(runs_dir: Path, task_id: str) -> list[RunRecord]:
    """Load latest + archived history runs for one task."""
    task_dir = resolve_task_run_dir(runs_dir, task_id)
    if task_dir is None:
        return []
    records: list[RunRecord] = []
    if (task_dir / "trace.json").is_file():
        records.append(load_run_record(task_dir, run_label="latest"))
    history = task_dir / "history"
    if history.is_dir():
        for archived in sorted(history.iterdir()):
            if archived.is_dir() and (archived / "trace.json").is_file():
                records.append(load_run_record(archived, run_label=archived.name))
    return records

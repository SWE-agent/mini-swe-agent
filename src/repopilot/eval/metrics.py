"""Aggregate metrics from normalized run records."""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict
from statistics import mean

from repopilot.eval.loader import RunRecord


def _pct(n: int, total: int) -> float:
    return round(100.0 * n / total, 1) if total else 0.0


def aggregate_runs(records: list[RunRecord]) -> dict:
    total = len(records)
    successes = sum(1 for r in records if r.outcome == "success")
    verify_pass = sum(1 for r in records if r.tests_passed is True)
    verify_fail = sum(1 for r in records if r.tests_passed is False)

    costs = [r.instance_cost for r in records]
    steps = [r.step_count for r in records]
    api_calls = [r.api_calls for r in records]
    repair_rounds = [r.repair_rounds for r in records]

    by_mode: dict[str, dict] = {}
    for mode in sorted({r.agent_mode for r in records}):
        subset = [r for r in records if r.agent_mode == mode]
        n = len(subset)
        by_mode[mode] = {
            "runs": n,
            "success_rate": _pct(sum(1 for r in subset if r.outcome == "success"), n),
            "verify_pass_rate": _pct(sum(1 for r in subset if r.tests_passed is True), n),
            "avg_cost": round(mean([r.instance_cost for r in subset]), 4) if subset else 0.0,
            "avg_steps": round(mean([r.step_count for r in subset]), 1) if subset else 0.0,
            "avg_api_calls": round(mean([r.api_calls for r in subset]), 1) if subset else 0.0,
        }

    return {
        "total_runs": total,
        "success_count": successes,
        "success_rate": _pct(successes, total),
        "verify_pass_count": verify_pass,
        "verify_pass_rate": _pct(verify_pass, total),
        "verify_fail_count": verify_fail,
        "avg_cost": round(mean(costs), 4) if costs else 0.0,
        "avg_steps": round(mean(steps), 1) if steps else 0.0,
        "avg_api_calls": round(mean(api_calls), 1) if api_calls else 0.0,
        "avg_repair_rounds": round(mean(repair_rounds), 1) if repair_rounds else 0.0,
        "by_agent_mode": by_mode,
    }


def failure_breakdown(records: list[RunRecord]) -> dict:
    failed = [r for r in records if r.outcome != "success"]
    by_category = Counter(r.failure_category or "unknown" for r in failed)
    by_stage = Counter(r.failure_stage or "unknown" for r in failed)

    return {
        "failed_runs": len(failed),
        "by_category": dict(sorted(by_category.items(), key=lambda x: (-x[1], x[0]))),
        "by_stage": dict(sorted(by_stage.items(), key=lambda x: (-x[1], x[0]))),
        "examples": [
            {
                "task_id": r.task_id,
                "agent_mode": r.agent_mode,
                "failure_category": r.failure_category,
                "failure_stage": r.failure_stage,
                "failed_step": r.failed_step,
                "failure_message": r.failure_message,
            }
            for r in failed
        ],
    }


def run_records_table(records: list[RunRecord]) -> list[dict]:
    return [
        {
            "task_id": r.task_id,
            "agent_mode": r.agent_mode,
            "outcome": r.outcome,
            "tests_passed": r.tests_passed,
            "failure_category": r.failure_category,
            "failure_stage": r.failure_stage,
            "failed_step": r.failed_step,
            "steps": r.step_count,
            "api_calls": r.api_calls,
            "cost": round(r.instance_cost, 4),
            "model": r.model,
        }
        for r in records
    ]


def record_to_dict(record: RunRecord) -> dict:
    return asdict(record)

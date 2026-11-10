import json

import pytest

from study_planner.loop import run_planning_loop
from study_planner.input import load_planner_input
from study_planner.loop import (
    run_planning_loop,
    run_planning_loop_from_data,
)


VALID_SCHEDULE = json.dumps(
    {
        "blocks": [
            {
                "task_id": "cs_midterm",
                "start": "2026-10-26T14:00:00",
                "end": "2026-10-26T16:00:00",
            }
        ]
    }
)


def test_loop_returns_valid_schedule():
    def generate(prompt):
        return VALID_SCHEDULE

    schedule = run_planning_loop(
        "eval_cases/simple_week.json",
        generate,
    )

    assert len(schedule.blocks) == 1
    assert schedule.blocks[0].task_id == "cs_midterm"


def test_loop_revises_invalid_schedule():
    prompts = []

    invalid_schedule = json.dumps(
        {
            "blocks": [
                {
                    "task_id": "cs_midterm",
                    "start": "2026-10-26T14:00:00",
                    "end": "2026-10-26T15:00:00",
                }
            ]
        }
    )

    responses = [
        invalid_schedule,
        VALID_SCHEDULE,
    ]

    def generate(prompt):
        prompts.append(prompt)
        return responses.pop(0)

    schedule = run_planning_loop(
        "eval_cases/simple_week.json",
        generate,
    )

    assert len(schedule.blocks) == 1
    assert len(prompts) == 2

    assert "failed deterministic validation" in prompts[1]
    assert "under-scheduled" in prompts[1]


def test_loop_revises_malformed_output():
    prompts = []

    responses = [
        "this is not json",
        VALID_SCHEDULE,
    ]

    def generate(prompt):
        prompts.append(prompt)
        return responses.pop(0)

    schedule = run_planning_loop(
        "eval_cases/simple_week.json",
        generate,
    )

    assert len(schedule.blocks) == 1
    assert len(prompts) == 2

    assert "could not be parsed" in prompts[1]


def test_loop_stops_after_iteration_limit():
    invalid_schedule = json.dumps(
        {
            "blocks": []
        }
    )

    def generate(prompt):
        return invalid_schedule

    with pytest.raises(
        RuntimeError,
        match="Unable to produce a valid schedule",
    ):
        run_planning_loop(
            "eval_cases/simple_week.json",
            generate,
            max_iterations=2,
        )

def test_loop_stops_when_workload_is_infeasible():
    called = False

    def generate(prompt):
        nonlocal called
        called = True
        return VALID_SCHEDULE

    with pytest.raises(
        ValueError,
        match="infeasible",
    ):
        run_planning_loop(
            "eval_cases/overloaded_week.json",
            generate,
        )

    assert not called

def test_loop_accepts_structured_input():
    tasks, availability = load_planner_input(
        "eval_cases/simple_week.json"
    )

    def generate(prompt):
        return VALID_SCHEDULE

    schedule = run_planning_loop_from_data(
        tasks,
        availability,
        generate,
    )

    assert len(schedule.blocks) == 1
    assert (
        schedule.blocks[0].task_id
        == "cs_midterm"
    )

def test_structured_loop_detects_infeasible_workload():
    tasks, availability = load_planner_input(
        "eval_cases/overloaded_week.json"
    )

    called = False

    def generate(prompt):
        nonlocal called
        called = True
        return VALID_SCHEDULE

    with pytest.raises(
        ValueError,
        match="infeasible",
    ):
        run_planning_loop_from_data(
            tasks,
            availability,
            generate,
        )

    assert not called
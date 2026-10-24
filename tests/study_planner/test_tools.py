from datetime import datetime

import pytest

from study_planner.schemas import Schedule, StudyBlock, Task
from study_planner.tools.deadlines import validate_deadlines

TASKS = [
    Task(id="cs_midterm", name="CS Midterm", deadline=datetime(2026, 10, 27, 12, 0), estimated_minutes=120, priority=1),
    Task(id="math_hw", name="Math HW", deadline=datetime(2026, 10, 30, 17, 0), estimated_minutes=60, priority=2),
]


def make_schedule(*blocks: tuple[str, datetime, datetime]) -> Schedule:
    return Schedule(blocks=[StudyBlock(task_id=t, start=s, end=e) for t, s, e in blocks])


@pytest.mark.parametrize(
    ("blocks",),
    [
        ([],),
        ([("cs_midterm", datetime(2026, 10, 26, 14, 0), datetime(2026, 10, 26, 16, 0))],),
        ([("cs_midterm", datetime(2026, 10, 27, 10, 0), datetime(2026, 10, 27, 12, 0))],),
        (
            [
                ("cs_midterm", datetime(2026, 10, 26, 9, 0), datetime(2026, 10, 26, 10, 0)),
                ("math_hw", datetime(2026, 10, 29, 9, 0), datetime(2026, 10, 29, 10, 0)),
            ],
        ),
    ],
)
def test_validate_deadlines_valid(blocks):
    result = validate_deadlines(make_schedule(*blocks), TASKS)
    assert result.validator == "deadline"
    assert result.valid
    assert result.issues == []


@pytest.mark.parametrize(
    ("start", "end"),
    [
        (datetime(2026, 10, 27, 11, 0), datetime(2026, 10, 27, 12, 1)),
        (datetime(2026, 10, 28, 9, 0), datetime(2026, 10, 28, 11, 0)),
    ],
)
def test_validate_deadlines_block_after_deadline(start, end):
    result = validate_deadlines(make_schedule(("cs_midterm", start, end)), TASKS)
    assert not result.valid
    assert len(result.issues) == 1
    assert (result.issues[0].task_id, result.issues[0].start, result.issues[0].end) == ("cs_midterm", start, end)
    assert "after deadline 2026-10-27T12:00:00" in result.issues[0].message


def test_validate_deadlines_reports_only_late_blocks():
    schedule = make_schedule(
        ("cs_midterm", datetime(2026, 10, 26, 9, 0), datetime(2026, 10, 26, 10, 0)),
        ("cs_midterm", datetime(2026, 10, 27, 13, 0), datetime(2026, 10, 27, 14, 0)),
        ("math_hw", datetime(2026, 10, 29, 9, 0), datetime(2026, 10, 29, 10, 0)),
        ("math_hw", datetime(2026, 10, 30, 16, 30), datetime(2026, 10, 30, 17, 30)),
    )
    result = validate_deadlines(schedule, TASKS)
    assert not result.valid
    assert [(i.task_id, i.start) for i in result.issues] == [
        ("cs_midterm", datetime(2026, 10, 27, 13, 0)),
        ("math_hw", datetime(2026, 10, 30, 16, 30)),
    ]


def test_validate_deadlines_unknown_task():
    result = validate_deadlines(
        make_schedule(("bio_review", datetime(2026, 10, 26, 9, 0), datetime(2026, 10, 26, 10, 0))), TASKS
    )
    assert not result.valid
    assert result.issues[0].task_id == "bio_review"
    assert "Unknown task" in result.issues[0].message


def test_validation_result_serializes():
    result = validate_deadlines(
        make_schedule(("cs_midterm", datetime(2026, 10, 28, 9, 0), datetime(2026, 10, 28, 10, 0))), TASKS
    )
    assert result.model_dump(mode="json") == {
        "validator": "deadline",
        "valid": False,
        "issues": [
            {
                "message": "Block for 'cs_midterm' ends at 2026-10-28T10:00:00, after deadline 2026-10-27T12:00:00",
                "task_id": "cs_midterm",
                "start": "2026-10-28T09:00:00",
                "end": "2026-10-28T10:00:00",
            }
        ],
    }

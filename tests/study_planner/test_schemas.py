from datetime import datetime

import pytest
from pydantic import ValidationError

from study_planner.schemas import (
    AvailabilityBlock,
    Schedule,
    StudyBlock,
    Task,
)


def test_valid_task():
    task = Task(
        id="cs_project",
        name="Finish project",
        course="CS 179M",
        deadline=datetime(2026, 10, 25, 23, 59),
        estimated_minutes=180,
        priority=3,
    )

    assert task.id == "cs_project"
    assert task.estimated_minutes == 180
    assert task.priority == 3


def test_task_rejects_nonpositive_estimated_minutes():
    with pytest.raises(ValidationError):
        Task(
            id="cs_project",
            name="Finish project",
            deadline=datetime(2026, 10, 25, 23, 59),
            estimated_minutes=0,
            priority=3,
        )


def test_task_rejects_invalid_priority():
    with pytest.raises(ValidationError):
        Task(
            id="cs_project",
            name="Finish project",
            deadline=datetime(2026, 10, 25, 23, 59),
            estimated_minutes=120,
            priority=5,
        )


def test_valid_availability_block():
    block = AvailabilityBlock(
        start=datetime(2026, 10, 20, 18, 0),
        end=datetime(2026, 10, 20, 21, 0),
    )

    assert block.end > block.start


def test_availability_rejects_invalid_range():
    with pytest.raises(ValidationError):
        AvailabilityBlock(
            start=datetime(2026, 10, 20, 21, 0),
            end=datetime(2026, 10, 20, 18, 0),
        )


def test_valid_study_block():
    block = StudyBlock(
        task_id="cs_project",
        start=datetime(2026, 10, 20, 18, 0),
        end=datetime(2026, 10, 20, 19, 0),
    )

    assert block.task_id == "cs_project"


def test_schedule_contains_blocks():
    schedule = Schedule(
        blocks=[
            StudyBlock(
                task_id="cs_project",
                start=datetime(2026, 10, 20, 18, 0),
                end=datetime(2026, 10, 20, 19, 0),
            )
        ]
    )

    assert len(schedule.blocks) == 1

def test_study_block_rejects_invalid_range():
    with pytest.raises(ValidationError):
        StudyBlock(
            task_id="cs_project",
            start=datetime(2026, 10, 20, 20, 0),
            end=datetime(2026, 10, 20, 18, 0),
        )
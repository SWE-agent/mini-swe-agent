import json
from datetime import datetime

import pytest

from study_planner.input import parse_availability, parse_tasks
from study_planner.schemas import AvailabilityBlock, Task


def test_parse_tasks_validates_json_into_shared_task_schema():
    tasks = parse_tasks(
        json.dumps(
            [
                {
                    "id": "cs_project",
                    "name": "Finish project",
                    "deadline": "2026-10-25T23:59:00",
                    "estimated_minutes": 180,
                    "priority": 3,
                }
            ]
        )
    )

    assert len(tasks) == 1
    assert isinstance(tasks[0], Task)
    assert tasks[0].deadline == datetime(2026, 10, 25, 23, 59)


def test_parse_tasks_rejects_invalid_json_and_schema():
    with pytest.raises(ValueError, match="Invalid task input"):
        parse_tasks("not json")
    with pytest.raises(ValueError, match="Invalid task input"):
        parse_tasks(json.dumps([{"id": "missing-required-fields"}]))


def test_parse_availability_validates_json_into_shared_schema():
    blocks = parse_availability(json.dumps([{"start": "2026-10-20T18:00:00", "end": "2026-10-20T21:00:00"}]))

    assert len(blocks) == 1
    assert isinstance(blocks[0], AvailabilityBlock)
    assert blocks[0].end > blocks[0].start


def test_parse_availability_rejects_invalid_json_and_time_range():
    with pytest.raises(ValueError, match="Invalid availability input"):
        parse_availability("not json")
    with pytest.raises(ValueError, match="Invalid availability input"):
        parse_availability(json.dumps([{"start": "2026-10-20T21:00:00", "end": "2026-10-20T18:00:00"}]))

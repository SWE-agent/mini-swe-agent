from datetime import datetime

import pytest

from study_planner.schemas import AvailabilityBlock, Schedule, StudyBlock
from study_planner.tools.availability import validate_availability

AVAILABILITY = [
    AvailabilityBlock(start=datetime(2026, 10, 26, 14, 0), end=datetime(2026, 10, 26, 16, 0)),
    AvailabilityBlock(start=datetime(2026, 10, 27, 9, 0), end=datetime(2026, 10, 27, 12, 0)),
]


@pytest.mark.parametrize(
    ("start", "end", "valid"),
    [
        (datetime(2026, 10, 26, 14, 30), datetime(2026, 10, 26, 15, 30), True),
        (datetime(2026, 10, 26, 14, 0), datetime(2026, 10, 26, 16, 0), True),
        (datetime(2026, 10, 27, 10, 0), datetime(2026, 10, 27, 12, 0), True),
        (datetime(2026, 10, 26, 13, 30), datetime(2026, 10, 26, 15, 0), False),
        (datetime(2026, 10, 26, 15, 0), datetime(2026, 10, 26, 16, 1), False),
        (datetime(2026, 10, 26, 18, 0), datetime(2026, 10, 26, 19, 0), False),
        (datetime(2026, 10, 26, 15, 0), datetime(2026, 10, 27, 10, 0), False),
    ],
)
def test_validate_availability(start, end, valid):
    result = validate_availability(Schedule(blocks=[StudyBlock(task_id="cs", start=start, end=end)]), AVAILABILITY)
    assert (result.validator, result.valid, len(result.issues)) == ("availability", valid, 0 if valid else 1)


def test_validate_availability_reports_only_bad_blocks():
    schedule = Schedule(
        blocks=[
            StudyBlock(task_id="cs", start=datetime(2026, 10, 26, 14, 0), end=datetime(2026, 10, 26, 15, 0)),
            StudyBlock(task_id="math", start=datetime(2026, 10, 26, 20, 0), end=datetime(2026, 10, 26, 21, 0)),
        ]
    )
    issues = validate_availability(schedule, AVAILABILITY).issues
    assert [(i.task_id, i.start) for i in issues] == [("math", datetime(2026, 10, 26, 20, 0))]
    assert "not inside any availability window" in issues[0].message

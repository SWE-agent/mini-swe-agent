from datetime import datetime

import pytest

from study_planner.schemas import Schedule, StudyBlock, Task
from study_planner.tools.completion import validate_completion


@pytest.mark.parametrize(("end_hour", "valid"), [(11, True), (10, False)])
def test_validate_completion(end_hour, valid):
    task = Task(id="cs", name="CS Midterm", deadline=datetime(2026, 10, 30), estimated_minutes=120, priority=1)
    block = StudyBlock(task_id="cs", start=datetime(2026, 10, 26, 9), end=datetime(2026, 10, 26, end_hour))
    result = validate_completion(Schedule(blocks=[block]), [task])
    assert (result.valid, len(result.issues)) == (valid, 0 if valid else 1)

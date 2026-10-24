import json
import pytest
from study_planner.schemas import Schedule, StudyBlock
from study_planner.output import parse_schedule, format_schedule

def test_parse_valid_schedule():
    output = json.dumps({
        "blocks": [
            {
                "task_id": "cs_midterm",
                "start": "2026-10-26T14:00:00",
                "end": "2026-10-26T15:30:00"
            }
        ]
    })

    schedule = parse_schedule(output)
    assert isinstance(schedule, Schedule)
    assert len(schedule.blocks) == 1
    assert schedule.blocks[0].task_id == "cs_midterm"

def test_parse_empty_schedule():
    output = json.dumps({"blocks": []})
    schedule = parse_schedule(output)
    assert isinstance(schedule, Schedule)
    assert len(schedule.blocks) == 0

def test_parse_invalid_json():
    output = "invalid json"
    with pytest.raises(ValueError, match="Invalid schedule output"):
        parse_schedule(output)

def test_parse_missing_fields():
    output = json.dumps({
        "blocks": [
            {
                "task_id": "cs_midterm",
                "start": "2026-10-26T14:00:00"
                # 'end' field is missing
            }
        ]
    })
    with pytest.raises(ValueError, match="Invalid schedule output"):
        parse_schedule(output)

def test_parse_invalid_time_range():
    output = json.dumps({
        "blocks": [
            {
                "task_id": "cs_midterm",
                "start": "2026-10-26T15:30:00",
                "end": "2026-10-26T14:00:00"  # end is before start
            }
        ]
    })
    with pytest.raises(ValueError, match="Invalid schedule output"):
        parse_schedule(output)

def test_format_schedule():
    schedule = Schedule(
        blocks=[
            StudyBlock(
                task_id="cs_midterm",
                start="2026-10-26T14:00:00",
                end="2026-10-26T15:30:00"
            )
        ]
    )
    formatted_output = format_schedule(schedule)
    expected_output = json.dumps({
        "blocks": [
            {
                "task_id": "cs_midterm",
                "start": "2026-10-26T14:00:00",
                "end": "2026-10-26T15:30:00"
            }
        ]
    }, indent=2)
    assert formatted_output == expected_output

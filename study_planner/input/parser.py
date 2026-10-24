import json

from pydantic import TypeAdapter, ValidationError

from study_planner.schemas import AvailabilityBlock, Task


def parse_tasks(raw_input: str) -> list[Task]:
    try:
        return TypeAdapter(list[Task]).validate_python(json.loads(raw_input))
    except (json.JSONDecodeError, ValidationError) as e:
        raise ValueError(f"Invalid task input: {e}") from e


def parse_availability(raw_input: str) -> list[AvailabilityBlock]:
    try:
        return TypeAdapter(list[AvailabilityBlock]).validate_python(json.loads(raw_input))
    except (json.JSONDecodeError, ValidationError) as e:
        raise ValueError(f"Invalid availability input: {e}") from e

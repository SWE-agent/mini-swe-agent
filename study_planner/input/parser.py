import json
from pathlib import Path

from pydantic import TypeAdapter, ValidationError

from study_planner.schemas import AvailabilityBlock, Task


def parse_tasks(raw_input: str) -> list[Task]:
    try:
        return TypeAdapter(list[Task]).validate_python(json.loads(raw_input))
    except (json.JSONDecodeError, ValidationError) as exc:
        raise ValueError(f"Invalid task input: {exc}") from exc


def parse_availability(raw_input: str) -> list[AvailabilityBlock]:
    try:
        return TypeAdapter(list[AvailabilityBlock]).validate_python(json.loads(raw_input))
    except (json.JSONDecodeError, ValidationError) as exc:
        raise ValueError(f"Invalid availability input: {exc}") from exc


def load_planner_input(
    path: str | Path,
) -> tuple[list[Task], list[AvailabilityBlock]]:
    path = Path(path)

    try:
        with path.open() as file:
            data = json.load(file)
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Unable to load planner input: {exc}") from exc

    if "tasks" not in data:
        raise ValueError("Planner input is missing 'tasks'")

    if "availability" not in data:
        raise ValueError("Planner input is missing 'availability'")

    try:
        tasks = [Task.model_validate(task) for task in data["tasks"]]

        availability = [AvailabilityBlock.model_validate(block) for block in data["availability"]]
    except ValidationError as exc:
        raise ValueError(f"Invalid planner input: {exc}") from exc

    return tasks, availability

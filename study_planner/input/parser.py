import json
from pathlib import Path

from pydantic import TypeAdapter, ValidationError

from study_planner.schemas import AvailabilityBlock, SchedulingPreferences, Task


def parse_tasks(raw_input: str) -> list[Task]:
    try:
        return TypeAdapter(list[Task]).validate_python(
            json.loads(raw_input)
        )
    except (json.JSONDecodeError, ValidationError) as exc:
        raise ValueError(f"Invalid task input: {exc}") from exc


def parse_availability(raw_input: str) -> list[AvailabilityBlock]:
    try:
        return TypeAdapter(list[AvailabilityBlock]).validate_python(
            json.loads(raw_input)
        )
    except (json.JSONDecodeError, ValidationError) as exc:
        raise ValueError(
            f"Invalid availability input: {exc}"
        ) from exc


def parse_planner_input(
    data: object,
) -> tuple[list[Task], list[AvailabilityBlock]]:
    if not isinstance(data, dict):
        raise ValueError("Planner input must be a JSON object")

    if "tasks" not in data:
        raise ValueError("Planner input is missing 'tasks'")

    if "availability" not in data:
        raise ValueError("Planner input is missing 'availability'")

    if not isinstance(data["tasks"], list):
        raise ValueError(
            "Planner input 'tasks' must be a list"
        )

    if not isinstance(data["availability"], list):
        raise ValueError(
            "Planner input 'availability' must be a list"
        )

    try:
        tasks = [
            Task.model_validate(task)
            for task in data["tasks"]
        ]

        availability = [
            AvailabilityBlock.model_validate(block)
            for block in data["availability"]
        ]
    except ValidationError as exc:
        raise ValueError(
            f"Invalid planner input: {exc}"
        ) from exc

    return tasks, availability


def parse_preferences(data: dict) -> SchedulingPreferences:
    try:
        return SchedulingPreferences.model_validate(
            data.get("preferences") or {}
        )
    except ValidationError as exc:
        raise ValueError(
            f"Invalid scheduling preferences: {exc}"
        ) from exc


def _load_json(path: str | Path) -> object:
    try:
        with Path(path).open() as file:
            return json.load(file)
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(
            f"Unable to load planner input: {exc}"
        ) from exc


def load_planner_input(
    path: str | Path,
) -> tuple[list[Task], list[AvailabilityBlock]]:
    return parse_planner_input(_load_json(path))


def load_preferences(path: str | Path) -> SchedulingPreferences:
    return parse_preferences(_load_json(path))
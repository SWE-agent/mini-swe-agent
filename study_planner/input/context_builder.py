import json

from study_planner.schemas import AvailabilityBlock, Task


def build_planner_context(
    tasks: list[Task],
    availability: list[AvailabilityBlock],
) -> dict[str, str]:
    task_data = [
        task.model_dump(mode="json")
        for task in tasks
    ]

    availability_data = [
        block.model_dump(mode="json")
        for block in availability
    ]

    return {
        "tasks": json.dumps(task_data, indent=2),
        "availability": json.dumps(availability_data, indent=2),
    }
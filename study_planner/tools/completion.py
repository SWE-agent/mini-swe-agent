from collections import Counter

from study_planner.schemas import Schedule, Task, ValidationIssue, ValidationResult


def validate_completion(schedule: Schedule, tasks: list[Task]) -> ValidationResult:
    scheduled = Counter()
    for block in schedule.blocks:
        scheduled[block.task_id] += int((block.end - block.start).total_seconds() // 60)
    issues = [
        ValidationIssue(
            message=(
                f"Task '{task.id}' is under-scheduled: {scheduled[task.id]} of {task.estimated_minutes} minutes "
                f"scheduled, {task.estimated_minutes - scheduled[task.id]} missing"
            ),
            task_id=task.id,
        )
        for task in tasks
        if scheduled[task.id] < task.estimated_minutes
    ]
    return ValidationResult(validator="completion", valid=not issues, issues=issues)

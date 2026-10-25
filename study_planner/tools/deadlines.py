from study_planner.schemas import Schedule, Task, ValidationIssue, ValidationResult


def validate_deadlines(schedule: Schedule, tasks: list[Task]) -> ValidationResult:
    deadlines = {task.id: task.deadline for task in tasks}
    issues = []
    for block in schedule.blocks:
        if block.task_id not in deadlines:
            message = f"Unknown task '{block.task_id}'"
        elif block.end > deadlines[block.task_id]:
            message = (
                f"Block for '{block.task_id}' ends at {block.end.isoformat()}, "
                f"after deadline {deadlines[block.task_id].isoformat()}"
            )
        else:
            continue
        issues.append(ValidationIssue(message=message, task_id=block.task_id, start=block.start, end=block.end))
    return ValidationResult(validator="deadline", valid=not issues, issues=issues)

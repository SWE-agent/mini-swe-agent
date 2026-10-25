from datetime import datetime

from study_planner.schemas import AvailabilityBlock, Task, ValidationIssue, ValidationResult


def available_minutes(availability: list[AvailabilityBlock], deadline: datetime) -> int:
    return sum(max(0, int((min(w.end, deadline) - w.start).total_seconds() // 60)) for w in availability)


def validate_workload(tasks: list[Task], availability: list[AvailabilityBlock]) -> ValidationResult:
    issues = []
    for deadline in sorted({task.deadline for task in tasks}):
        due = [task for task in tasks if task.deadline <= deadline]
        required = sum(task.estimated_minutes for task in due)
        available = available_minutes(availability, deadline)
        if required > available:
            issues.append(
                ValidationIssue(
                    message=(
                        f"Infeasible: {required} minutes of work ({', '.join(t.id for t in due)}) due by "
                        f"{deadline.isoformat()}, but only {available} minutes available before then"
                    )
                )
            )
    return ValidationResult(validator="workload", valid=not issues, issues=issues)

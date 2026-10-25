from study_planner.schemas import AvailabilityBlock, Schedule, ValidationIssue, ValidationResult


def validate_availability(schedule: Schedule, availability: list[AvailabilityBlock]) -> ValidationResult:
    issues = [
        ValidationIssue(
            message=(
                f"Block for '{block.task_id}' from {block.start.isoformat()} to {block.end.isoformat()} "
                "is not inside any availability window"
            ),
            task_id=block.task_id,
            start=block.start,
            end=block.end,
        )
        for block in schedule.blocks
        if not any(window.start <= block.start and block.end <= window.end for window in availability)
    ]
    return ValidationResult(validator="availability", valid=not issues, issues=issues)

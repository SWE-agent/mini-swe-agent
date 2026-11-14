from study_planner.schemas import (
    Schedule,
    ValidationIssue,
    ValidationResult,
)


def validate_overlaps(
    schedule: Schedule,
) -> ValidationResult:
    blocks = sorted(
        schedule.blocks,
        key=lambda block: block.start,
    )

    issues = []

    for i, current in enumerate(blocks):
        for next_block in blocks[i + 1 :]:
            if next_block.start >= current.end:
                break

            issues.append(
                ValidationIssue(
                    message=(
                        f"Block for '{current.task_id}' from "
                        f"{current.start.isoformat()} to "
                        f"{current.end.isoformat()} overlaps "
                        f"with block for '{next_block.task_id}' "
                        f"from {next_block.start.isoformat()} to "
                        f"{next_block.end.isoformat()}"
                    ),
                    task_id=next_block.task_id,
                    start=next_block.start,
                    end=next_block.end,
                )
            )

    return ValidationResult(
        validator="overlap",
        valid=not issues,
        issues=issues,
    )

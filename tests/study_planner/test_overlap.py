from datetime import datetime

from study_planner.schemas import Schedule, StudyBlock
from study_planner.tools.overlap import validate_overlaps


def make_schedule(*blocks):
    return Schedule(
        blocks=[
            StudyBlock(
                task_id=task_id,
                start=start,
                end=end,
            )
            for task_id, start, end in blocks
        ]
    )


def test_no_blocks_is_valid():
    result = validate_overlaps(Schedule(blocks=[]))

    assert result.validator == "overlap"
    assert result.valid
    assert result.issues == []


def test_single_block_is_valid():
    schedule = make_schedule(
        (
            "task_a",
            datetime(2026, 10, 26, 14, 0),
            datetime(2026, 10, 26, 15, 0),
        )
    )

    result = validate_overlaps(schedule)

    assert result.valid


def test_adjacent_blocks_do_not_overlap():
    schedule = make_schedule(
        (
            "task_a",
            datetime(2026, 10, 26, 14, 0),
            datetime(2026, 10, 26, 15, 0),
        ),
        (
            "task_b",
            datetime(2026, 10, 26, 15, 0),
            datetime(2026, 10, 26, 16, 0),
        ),
    )

    result = validate_overlaps(schedule)

    assert result.valid


def test_partial_overlap_is_invalid():
    schedule = make_schedule(
        (
            "task_a",
            datetime(2026, 10, 26, 14, 0),
            datetime(2026, 10, 26, 15, 30),
        ),
        (
            "task_b",
            datetime(2026, 10, 26, 15, 0),
            datetime(2026, 10, 26, 16, 0),
        ),
    )

    result = validate_overlaps(schedule)

    assert not result.valid
    assert len(result.issues) == 1
    assert "overlap" in result.issues[0].message.lower()


def test_nested_block_is_invalid():
    schedule = make_schedule(
        (
            "task_a",
            datetime(2026, 10, 26, 14, 0),
            datetime(2026, 10, 26, 17, 0),
        ),
        (
            "task_b",
            datetime(2026, 10, 26, 15, 0),
            datetime(2026, 10, 26, 16, 0),
        ),
    )

    result = validate_overlaps(schedule)

    assert not result.valid
    assert len(result.issues) == 1


def test_one_long_block_overlaps_multiple_blocks():
    schedule = make_schedule(
        (
            "task_a",
            datetime(2026, 10, 26, 14, 0),
            datetime(2026, 10, 26, 18, 0),
        ),
        (
            "task_b",
            datetime(2026, 10, 26, 15, 0),
            datetime(2026, 10, 26, 16, 0),
        ),
        (
            "task_c",
            datetime(2026, 10, 26, 17, 0),
            datetime(2026, 10, 26, 17, 30),
        ),
    )

    result = validate_overlaps(schedule)

    assert not result.valid
    assert len(result.issues) == 2

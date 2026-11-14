from collections.abc import Callable
from pathlib import Path

from study_planner.agent import (
    build_agent_prompt,
    build_agent_prompt_from_data,
)

from study_planner.input import load_planner_input, load_preferences
from study_planner.output import parse_schedule
from study_planner.schemas import (AvailabilityBlock, Schedule, SchedulingPreferences, Task, ValidationResult)
from study_planner.tools.availability import validate_availability
from study_planner.tools.completion import validate_completion
from study_planner.tools.deadlines import validate_deadlines
from study_planner.tools.workload import validate_workload
from study_planner.tools.overlap import validate_overlaps


GenerateSchedule = Callable[[str], str]
RecordEvent = Callable[[str], None]


def validate_schedule(schedule: Schedule, tasks: list[Task], availability: list[AvailabilityBlock]) -> list[ValidationResult]:
  return [validate_deadlines(schedule, tasks), validate_availability(schedule, availability),validate_completion(schedule, tasks), validate_overlaps(schedule)]


def format_validation_feedback(results: list[ValidationResult]) -> str:
    messages = []

    for result in results:
        if result.valid:
            continue

        for issue in result.issues:
            messages.append(
                f"[{result.validator}] {issue.message}"
            )

    return "\n".join(messages)


def run_planning_loop(
    input_path: str | Path,
    generate: GenerateSchedule,
    max_iterations: int = 5,
    record_event: RecordEvent | None = None,
) -> Schedule:
    tasks, availability = load_planner_input(
        input_path
    )

    return run_planning_loop_from_data(
        tasks,
        availability,
        generate,
        max_iterations=max_iterations,
        record_event=record_event,
        preferences=load_preferences(input_path),
    )

def run_planning_loop_from_data(
    tasks: list[Task],
    availability: list[AvailabilityBlock],
    generate: GenerateSchedule,
    max_iterations: int = 5,
    record_event: RecordEvent | None = None,
    preferences: SchedulingPreferences | None = None,
) -> Schedule:
    workload_result = validate_workload(
        tasks,
        availability,
    )

    if not workload_result.valid:
        feedback = format_validation_feedback(
            [workload_result]
        )

        raise ValueError(
            f"Study plan is infeasible:\n{feedback}"
        )

    initial_prompt = build_agent_prompt_from_data(
        tasks,
        availability,
        preferences,
    )

    prompt = initial_prompt

    for _ in range(max_iterations):
        raw_output = generate(prompt)

        try:
            schedule = parse_schedule(raw_output)

        except ValueError as exc:
            if record_event:
                record_event("format_failure")

            prompt = (
                f"{initial_prompt}\n\n"
                "Your previous response could not be parsed "
                "as a valid schedule.\n\n"
                f"Parser feedback:\n{exc}\n\n"
                "Return only a corrected schedule using the "
                "required JSON format."
            )

            continue

        validation_results = validate_schedule(
            schedule,
            tasks,
            availability,
        )

        if record_event:
            for result in validation_results:
                if not result.valid:
                    record_event(
                        "validator_failure"
                    )

        if all(
            result.valid
            for result in validation_results
        ):
            return schedule

        feedback = format_validation_feedback(
            validation_results
        )

        prompt = (
            f"{initial_prompt}\n\n"
            "The previous schedule failed deterministic "
            "validation.\n\n"
            f"Validation feedback:\n{feedback}\n\n"
            "Revise the schedule to fix every validation "
            "error and return the corrected schedule."
        )

    raise RuntimeError(
        "Unable to produce a valid schedule within "
        f"{max_iterations} iterations"
    )
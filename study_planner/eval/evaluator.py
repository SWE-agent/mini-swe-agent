import json
from collections.abc import Callable
from pathlib import Path

from study_planner.eval.metrics import calculate_metrics
from study_planner.loop import run_planning_loop

GenerateSchedule = Callable[[str], str]


def load_eval_cases(cases_dir: str | Path | None = None) -> list[tuple[Path, dict]]:
    if cases_dir is None:
        cases_dir = Path(__file__).resolve().parents[2] / "eval_cases"

    cases_dir = Path(cases_dir)
    cases = []

    for path in sorted(cases_dir.glob("*.json")):
        with path.open(encoding="utf-8") as file:
            case = json.load(file)

        if "id" not in case:
            raise ValueError(f"{path.name}: missing id")

        if "tasks" not in case or "availability" not in case:
            raise ValueError(f"{path.name}: missing input data")

        if "expected_output" not in case:
            raise ValueError(f"{path.name}: missing expected_output")

        if not isinstance(case["expected_output"].get("feasible"), bool):
            raise ValueError(f"{path.name}: expected_output.feasible must be boolean")

        cases.append((path, case))

    if not cases:
        raise ValueError(f"No evaluation cases found in {cases_dir}")

    return cases


def evaluate_cases(generate: GenerateSchedule, cases_dir: str | Path | None = None) -> list[dict]:
    results = []

    for path, case in load_eval_cases(cases_dir):
        expected_feasible = case["expected_output"]["feasible"]

        attempts = 0
        validator_failures = 0
        format_failures = 0
        tool_call_failures = 0

        def record_event(event: str) -> None:
            nonlocal validator_failures, format_failures

            if event == "validator_failure":
                validator_failures += 1
            elif event == "format_failure":
                format_failures += 1

        def tracked_generate(prompt: str) -> str:
            nonlocal attempts, tool_call_failures
            attempts += 1

            try:
                return generate(prompt)
            except Exception:
                tool_call_failures += 1
                raise

        schedule_valid = False
        predicted_feasible = None
        error = None

        try:
            run_planning_loop(
                path,
                tracked_generate,
                record_event=record_event,
            )
            schedule_valid = True
            predicted_feasible = True

        except ValueError as exc:
            error = str(exc)

            if "infeasible" in error.lower():
                predicted_feasible = False

        except Exception as exc:
            error = str(exc)

        results.append(
            {
                "case_id": case["id"],
                "expected_feasible": expected_feasible,
                "schedule_valid": schedule_valid,
                "predicted_feasible": predicted_feasible,
                "revisions": max(0, attempts - 1),
                "validator_failures": validator_failures,
                "format_failures": format_failures,
                "tool_call_failures": tool_call_failures,
                "error": error,
            }
        )
    return results


def evaluate(generate, cases_dir=None):
    results = evaluate_cases(generate, cases_dir)
    return calculate_metrics(results)

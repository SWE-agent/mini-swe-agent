def calculate_metrics(results: list[dict]) -> dict:
    total_runs = len(results)

    feasible_runs = [result for result in results if result["expected_feasible"]]

    infeasible_runs = [result for result in results if not result["expected_feasible"]]

    valid_schedules = sum(result["schedule_valid"] for result in feasible_runs)

    correctly_detected = sum(result["predicted_feasible"] is False for result in infeasible_runs)

    return {
        "total_runs": total_runs,
        "valid_schedule_rate": (valid_schedules / len(feasible_runs) if feasible_runs else None),
        "average_revisions": (sum(result["revisions"] for result in results) / total_runs if total_runs else None),
        "validator_failures": sum(result["validator_failures"] for result in results),
        "infeasible_detection_rate": (correctly_detected / len(infeasible_runs) if infeasible_runs else None),
        "tool_call_failures": sum(result["tool_call_failures"] for result in results),
        "format_failures": sum(result["format_failures"] for result in results),
    }

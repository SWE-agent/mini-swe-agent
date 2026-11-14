import json
from pathlib import Path

from study_planner.eval import calculate_metrics, evaluate_cases
from study_planner.model import generate_schedule

OUTPUT_PATH = Path("study_planner/eval/results/mvp_results.json")


def main():
    results = evaluate_cases(
        generate_schedule,
        cases_dir="eval_cases",
    )

    metrics = calculate_metrics(results)

    output = {
        "results": results,
        "metrics": metrics,
    }

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    OUTPUT_PATH.write_text(json.dumps(output, indent=2) + "\n")

    print("\n=== PER-CASE RESULTS ===")
    print(json.dumps(results, indent=2))

    print("\n=== MVP METRICS ===")
    print(json.dumps(metrics, indent=2))

    print(f"\nSaved evaluation results to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()

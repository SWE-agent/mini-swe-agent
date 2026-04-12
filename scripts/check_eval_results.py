"""Check SWE-bench evaluation results from report.json files."""
import json
import os
import sys


def check_results(eval_dir):
    results = []

    for instance_id in sorted(os.listdir(eval_dir)):
        report_path = os.path.join(eval_dir, instance_id, "report.json")
        if not os.path.isfile(report_path):
            continue

        report = json.loads(open(report_path).read())
        info = report[instance_id]

        resolved = info.get("resolved", False)
        patch_applied = info.get("patch_successfully_applied", False)
        f2p = info.get("tests_status", {}).get("FAIL_TO_PASS", {})
        p2p = info.get("tests_status", {}).get("PASS_TO_PASS", {})
        p2f = info.get("tests_status", {}).get("PASS_TO_FAIL", {})

        f2p_success = len(f2p.get("success", []))
        f2p_failure = len(f2p.get("failure", []))
        p2p_failure = len(p2p.get("failure", []))
        p2f_failure = len(p2f.get("failure", []))

        results.append({
            "id": instance_id,
            "resolved": resolved,
            "patch_applied": patch_applied,
            "bugs_fixed": f2p_success,
            "bugs_not_fixed": f2p_failure,
            "regressions": p2p_failure + p2f_failure,
        })

    # Print results
    resolved_count = sum(1 for r in results if r["resolved"])
    total = len(results)

    print(f"{'Instance ID':<45} {'Resolved':<10} {'Fixed':<7} {'Not Fixed':<10} {'Regression'}")
    print("-" * 90)
    if total == 0:
        print("No report.json files found in the given directory.")
        print("-" * 90)
        return

    for r in results:
        status = "PASS" if r["resolved"] else "FAIL"
        print(f"{r['id']:<45} {status:<10} {r['bugs_fixed']:<7} {r['bugs_not_fixed']:<10} {r['regressions']}")

    print("-" * 90)
    print(f"Total: {total}  |  Resolved: {resolved_count}  |  Failed: {total - resolved_count}  |  Rate: {resolved_count/total*100:.1f}%")


if __name__ == "__main__":
    eval_dir = sys.argv[1] if len(sys.argv) > 1 else "logs/run_evaluation/deepseek_bench_eval/deepseek__deepseek-chat"
    if not os.path.isdir(eval_dir):
        print(f"Directory not found: {eval_dir}")
        sys.exit(1)
    check_results(eval_dir)

"""Compare PlanMem run to the upstream baseline run on SWE-bench Verified.

Baseline reference: experiments/qwen3_baseline_verified/ + the sb-cli report at
sb-cli-reports/swe-bench_verified__test__qwen3_coder_baseline_rerun_complete.json
(146/500 resolved = 29.2%, recorded in experiments/BASELINE_RECORD.md).

Reports:
- Resolved-count delta
- Per-instance delta: which instances PlanMem fixes that baseline doesn't, and
  which it loses
- Categorical attribution: gain/loss bucketed by whether the patch differs

Usage:
    python -m experiments.eval.compare_to_baseline \\
        --planmem sb-cli-reports/<planmem_report>.json \\
        --baseline sb-cli-reports/swe-bench_verified__test__qwen3_coder_baseline_rerun_complete.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def load(path: str) -> dict:
    return json.loads(Path(path).read_text())


def resolved_set(report: dict) -> set[str]:
    """sb-cli report shape: {'resolved': [ids], 'unresolved': [ids], ...}"""
    r = report.get("resolved")
    if isinstance(r, list):
        return set(r)
    if isinstance(r, dict):
        return set(r.keys())
    # alt: per-instance map
    inst = report.get("instances") or {}
    return {iid for iid, v in inst.items() if v.get("resolved")}


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--planmem", required=True)
    p.add_argument("--baseline", required=True)
    args = p.parse_args()

    pm = load(args.planmem)
    bl = load(args.baseline)
    pm_res = resolved_set(pm)
    bl_res = resolved_set(bl)

    only_pm = sorted(pm_res - bl_res)
    only_bl = sorted(bl_res - pm_res)
    both = pm_res & bl_res

    print(f"baseline resolved: {len(bl_res)}")
    print(f"planmem  resolved: {len(pm_res)}")
    print(f"both           : {len(both)}")
    print(f"planmem-only   : {len(only_pm)}")
    print(f"baseline-only  : {len(only_bl)}")
    print(f"net delta       : {len(pm_res) - len(bl_res):+d}")
    print()
    if only_pm:
        print("=== planmem fixes (baseline did NOT) — diagnostic of PlanMem wins ===")
        for iid in only_pm[:30]:
            print(f"  + {iid}")
    if only_bl:
        print("\n=== baseline fixes (planmem did NOT) — diagnostic of PlanMem regressions ===")
        for iid in only_bl[:30]:
            print(f"  - {iid}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Deep analysis of PlanMem trajectories on SWE-bench.

Run AFTER a batch run completes. Computes:
- Exit-status distribution (Submitted vs LimitsExceeded vs error)
- Per-instance: n_calls, patch_len, planner progress
- Phase distribution across the run (how often EXPLORATION/HYPOTHESIS/...)
- Replan / drift / backtrack firing rates
- Metadata coverage (toolcall extra.actions vs bash-fence parse)
- Sub-task completion vs failure rates

Use this to triage why PlanMem under-performs (or doesn't beat) baseline.

Usage:
    python -m experiments.eval.analyze_trajectories \\
        --traj-dir experiments/outputs/planmem_full
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path

# ── Per-trajectory parsers ──────────────────────────────────────────────────


def load_traj(path: Path) -> dict:
    return json.loads(path.read_text())


def assistant_messages(traj: dict) -> list[dict]:
    return [m for m in traj.get("messages", []) if m.get("role") == "assistant"]


def observation_messages(traj: dict) -> list[dict]:
    return [
        m
        for m in traj.get("messages", [])
        if m.get("role") in ("user", "tool") and "<returncode>" in str(m.get("content", ""))
    ]


def extract_command(asst_msg: dict) -> str:
    """Pull the command out of an assistant message in any model mode."""
    actions = (asst_msg.get("extra") or {}).get("actions") or []
    if actions:
        a0 = actions[0]
        if isinstance(a0, dict):
            for k in ("command", "action", "cmd"):
                if a0.get(k):
                    return a0[k].strip()
            args = a0.get("arguments") or {}
            if isinstance(args, dict):
                for k in ("command", "cmd"):
                    if args.get(k):
                        return args[k].strip()
        elif isinstance(a0, str):
            return a0.strip()
    text = asst_msg.get("content") or ""
    m = re.search(r"```(?:mswea_bash_command|bash)\s*\n(.*?)\n```", text, re.DOTALL)
    return m.group(1).strip() if m else ""


# Lightweight phase detection mirroring planmem.phase_detector logic, but
# self-contained so we can analyze any traj without importing PlanMem.
_IMPL = re.compile(
    r"^\s*(sed|awk|perl|ed|ex|tee|cp|mv|truncate|apply_patch|patch|"
    r"git\s+(apply|checkout|restore|mv|rm))\b",
    re.IGNORECASE,
)
_IMPL_REDIRECT = re.compile(r"(^|\s)(>|>>)\s*\S")
_IMPL_HEREDOC = re.compile(r"cat\s+<<")
_VERIF = re.compile(
    r"^\s*(python|python3|pytest|py\.test|tox|nox|make\s+test|"
    r"npm\s+test|yarn\s+test|go\s+test|cargo\s+test|"
    r"\.\/test|bash\s+test|unittest)\b",
    re.IGNORECASE,
)
_HYP_CMD = re.compile(
    r"^\s*(python\d?\s+-c|python\d?\s+-m\s+pdb|pdb|ipython|"
    r"git\s+blame|git\s+log\s+-p)\b",
    re.IGNORECASE,
)
_EXPL = re.compile(
    r"^\s*(find|grep|rg|ag|ack|ls|tree|cat|head|tail|less|more|wc|file|"
    r"nl|bat|fd|locate|git\s+(log|show|diff|blame))\b",
    re.IGNORECASE,
)


def detect_phase(cmd: str) -> str:
    first = cmd.split("\n", 1)[0].strip() if cmd else ""
    if _IMPL.search(first) or _IMPL_REDIRECT.search(first) or _IMPL_HEREDOC.search(first):
        return "implementation"
    if _HYP_CMD.search(first):
        return "hypothesis"
    if _VERIF.search(first):
        return "verification"
    if _EXPL.search(first):
        return "exploration"
    return "other"


# ── Per-traj metrics ────────────────────────────────────────────────────────


def analyze_traj(traj: dict, instance_id: str) -> dict:
    info = traj.get("info") or {}
    asst = assistant_messages(traj)
    obs = observation_messages(traj)
    n_calls = len(asst)
    n_obs = len(obs)
    submission = info.get("submission") or ""
    exit_status = info.get("exit_status") or ""

    # Phase distribution from assistant commands.
    phases: Counter = Counter()
    metadata_hits = 0  # observations whose preceding asst had a parseable command
    nonzero_rc = 0
    for a in asst:
        cmd = extract_command(a)
        if cmd:
            metadata_hits += 1
            phases[detect_phase(cmd)] += 1
        else:
            phases["unknown"] += 1
    # Return-code statistics on observations.
    for o in obs:
        m = re.search(r"<returncode>(\d+)</returncode>", str(o.get("content", "")))
        if m and int(m.group(1)) != 0:
            nonzero_rc += 1

    planner_progress = traj.get("planner_progress") or ""
    # Parse "N/M sub-tasks done, K failed, phase=..."
    done = total = failed = None
    pm = re.search(r"(\d+)/(\d+)\s+sub-tasks?\s+done,\s+(\d+)\s+failed", planner_progress)
    if pm:
        done, total, failed = int(pm.group(1)), int(pm.group(2)), int(pm.group(3))

    return {
        "instance_id": instance_id,
        "exit_status": exit_status,
        "submitted": exit_status == "Submitted",
        "patch_len": len(submission),
        "patch_empty": (len(submission) == 0),
        "n_calls": n_calls,
        "n_obs": n_obs,
        "metadata_hit_rate": metadata_hits / max(1, n_calls),
        "nonzero_rc_rate": nonzero_rc / max(1, n_obs),
        "phases": dict(phases),
        "planner_done": done,
        "planner_total": total,
        "planner_failed": failed,
    }


# ── Aggregate ───────────────────────────────────────────────────────────────


def aggregate(rows: list[dict]) -> dict:
    n = len(rows)
    if n == 0:
        return {"n": 0}
    submitted = sum(r["submitted"] for r in rows)
    empty = sum(r["patch_empty"] for r in rows)
    total_phases: Counter = Counter()
    for r in rows:
        for p, c in r["phases"].items():
            total_phases[p] += c
    avg_calls = sum(r["n_calls"] for r in rows) / n
    avg_meta = sum(r["metadata_hit_rate"] for r in rows) / n
    avg_rc = sum(r["nonzero_rc_rate"] for r in rows) / n
    avg_done = sum((r["planner_done"] or 0) for r in rows) / n
    avg_failed = sum((r["planner_failed"] or 0) for r in rows) / n
    avg_total = sum((r["planner_total"] or 0) for r in rows) / n
    exit_dist: Counter = Counter(r["exit_status"] for r in rows)
    return {
        "n": n,
        "submitted_rate": submitted / n,
        "empty_patch_rate": empty / n,
        "avg_n_calls": avg_calls,
        "avg_metadata_hit_rate": avg_meta,
        "avg_nonzero_rc_rate": avg_rc,
        "avg_planner_done": avg_done,
        "avg_planner_failed": avg_failed,
        "avg_planner_total": avg_total,
        "phase_total": dict(total_phases),
        "exit_status_dist": dict(exit_dist),
    }


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--traj-dir", required=True, help="Directory containing <instance>/<instance>.traj.json files")
    args = p.parse_args()

    root = Path(args.traj_dir)
    if not root.is_dir():
        print(f"ERROR: {root} not a directory")
        return 2

    traj_paths = sorted(root.glob("*/*.traj.json"))
    print(f"Found {len(traj_paths)} trajectories under {root}\n")

    rows = []
    for tp in traj_paths:
        try:
            t = load_traj(tp)
            rows.append(analyze_traj(t, instance_id=tp.parent.name))
        except Exception as e:  # noqa: BLE001
            print(f"  skip {tp}: {e}")

    agg = aggregate(rows)
    print("=== aggregate ===")
    print(json.dumps(agg, indent=2, sort_keys=True))

    # Top empty-patch instances
    empty_instances = [r["instance_id"] for r in rows if r["patch_empty"]][:20]
    if empty_instances:
        print(f"\nempty-patch instances (first 20 of {sum(r['patch_empty'] for r in rows)}):")
        for iid in empty_instances:
            print(f"  - {iid}")

    # Save per-instance rows for downstream charting
    out = root / "trajectory_analysis.json"
    out.write_text(json.dumps({"aggregate": agg, "per_instance": rows}, indent=2))
    print(f"\nWrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

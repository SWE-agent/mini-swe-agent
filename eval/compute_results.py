#!/usr/bin/env python3
"""Compute evaluation metrics from mini-swe-agent trajectory files.

Reads trajectory JSONs and preds.json from an output directory and computes:
- Resolve rate (% solved) — requires SWE-bench evaluation harness
- Mean total tokens per task
- Mean wall-clock time per task
- Mean repair rounds (PR run only)
- Distribution of repair rounds

Usage:
    python eval/compute_results.py eval/output/baseline --output eval/results/baseline.json
    python eval/compute_results.py eval/output/pr_enabled --output eval/results/pr_enabled.json --patch-repair
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import typer

app = typer.Typer()


def _load_trajectories(output_dir: Path) -> list[dict[str, Any]]:
    """Load all trajectory JSON files from the output directory."""
    trajs = []
    for traj_path in sorted(output_dir.glob("*/*.traj.json")):
        try:
            trajs.append(json.loads(traj_path.read_text(encoding="utf-8")))
        except (json.JSONDecodeError, OSError) as e:
            print(f"  WARNING: Failed to load {traj_path}: {e}")
    return trajs


def _extract_tokens(traj: dict[str, Any]) -> dict[str, int]:
    """Extract prompt and completion token counts from trajectory messages."""
    prompt_tokens = 0
    completion_tokens = 0
    for msg in traj.get("messages", []):
        extra = msg.get("extra", {})
        response = extra.get("response", {})
        usage = response.get("usage", {}) if isinstance(response, dict) else {}
        prompt_tokens += usage.get("prompt_tokens", 0) or 0
        completion_tokens += usage.get("completion_tokens", 0) or 0
    return {"prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens}


def _extract_wall_clock(traj: dict[str, Any]) -> float | None:
    """Extract wall-clock time in seconds from trajectory messages."""
    messages = traj.get("messages", [])
    timestamps = []
    for msg in messages:
        ts = msg.get("extra", {}).get("timestamp")
        if ts is not None:
            timestamps.append(ts)
    if len(timestamps) >= 2:
        return timestamps[-1] - timestamps[0]
    return None


def _extract_patch_repair_info(traj: dict[str, Any]) -> dict[str, Any] | None:
    """Extract patch repair metadata from trajectory info."""
    info = traj.get("info", {})
    return info.get("patch_repair")


def compute(output_dir: Path, patch_repair_enabled: bool = False) -> dict[str, Any]:
    """Compute aggregate metrics from all trajectories in *output_dir*."""
    trajs = _load_trajectories(output_dir)
    if not trajs:
        return {"error": f"No trajectory files found in {output_dir}"}

    n = len(trajs)
    completed = 0
    repaired = 0
    total_prompt_tokens = 0
    total_completion_tokens = 0
    wall_clocks: list[float] = []
    repair_rounds: list[int] = []
    exit_statuses: dict[str, int] = {}

    for traj in trajs:
        info = traj.get("info", {})
        exit_status = info.get("exit_status", "unknown")
        exit_statuses[exit_status] = exit_statuses.get(exit_status, 0) + 1

        tokens = _extract_tokens(traj)
        total_prompt_tokens += tokens["prompt_tokens"]
        total_completion_tokens += tokens["completion_tokens"]

        wc = _extract_wall_clock(traj)
        if wc is not None:
            wall_clocks.append(wc)

        if patch_repair_enabled:
            pr_info = _extract_patch_repair_info(traj)
            if pr_info:
                repair_rounds.append(pr_info.get("rounds_used", 0))
                if pr_info.get("success"):
                    repaired += 1

        if exit_status == "submitted" or exit_status == "completed" or exit_status == "repaired":
            completed += 1

    result: dict[str, Any] = {
        "n_tasks": n,
        "n_completed": completed,
        "exit_statuses": exit_statuses,
        "total_tokens": {
            "prompt": total_prompt_tokens,
            "completion": total_completion_tokens,
            "total": total_prompt_tokens + total_completion_tokens,
        },
        "mean_tokens_per_task": {
            "prompt": round(total_prompt_tokens / n) if n else 0,
            "completion": round(total_completion_tokens / n) if n else 0,
            "total": round((total_prompt_tokens + total_completion_tokens) / n) if n else 0,
        },
    }

    if wall_clocks:
        result["mean_wall_clock_s"] = round(sum(wall_clocks) / len(wall_clocks))
        result["wall_clock_n"] = len(wall_clocks)

    if patch_repair_enabled and repair_rounds:
        result["patch_repair"] = {
            "mean_rounds": round(sum(repair_rounds) / len(repair_rounds), 2),
            "distribution": {
                "0_rounds": repair_rounds.count(0),
                "1_round": repair_rounds.count(1),
                "2_rounds": repair_rounds.count(2),
            },
            "n_repaired": repaired,
            "n_with_repair_data": len(repair_rounds),
        }

    result["generated_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    result["patch_repair_enabled"] = patch_repair_enabled

    return result


@app.command()
def main(
    output_dir: Path = typer.Argument(..., help="Path to the evaluation output directory"),
    result_file: Path = typer.Option(..., "--output", "-o", help="Path to save the results JSON"),
    patch_repair: bool = typer.Option(False, "--patch-repair", help="Extract patch repair metrics"),
) -> None:
    """Compute evaluation metrics from trajectory files."""
    result = compute(output_dir, patch_repair_enabled=patch_repair)
    result_file.parent.mkdir(parents=True, exist_ok=True)
    result_file.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Results saved to {result_file}")
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    app()

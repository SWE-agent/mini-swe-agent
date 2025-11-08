#!/usr/bin/env python3
"""Analyze and compare baseline vs memory experiment results."""

import argparse
import json
from pathlib import Path


def load_results(results_dir: Path) -> dict:
    """Load results from experiment directory.

    Args:
        results_dir: Path to results directory

    Returns:
        Dict with results data
    """
    results = {"trajectories": [], "metrics": {}, "metadata": {}}

    # Load trajectories
    traj_dir = results_dir / "trajectories"
    if traj_dir.exists():
        for traj_file in traj_dir.glob("*.json"):
            with open(traj_file) as f:
                results["trajectories"].append(json.load(f))

    # Load metrics
    metrics_file = results_dir / "metrics.json"
    if metrics_file.exists():
        with open(metrics_file) as f:
            results["metrics"] = json.load(f)

    # Load metadata
    metadata_file = results_dir / "metadata.json"
    if metadata_file.exists():
        with open(metadata_file) as f:
            results["metadata"] = json.load(f)

    return results


def compute_statistics(results: dict) -> dict:
    """Compute statistics from results.

    Args:
        results: Results dict

    Returns:
        Dict with statistics
    """
    trajectories = results["trajectories"]

    if not trajectories:
        return {}

    stats = {
        "total_instances": len(trajectories),
        "resolved": 0,
        "total_steps": 0,
        "total_cost": 0.0,
        "total_tokens": 0,
        "total_time": 0.0,
        "avg_steps": 0.0,
        "avg_cost": 0.0,
        "avg_tokens": 0.0,
        "avg_time": 0.0,
    }

    for traj in trajectories:
        # Resolution
        if traj.get("resolved", False):
            stats["resolved"] += 1

        # Steps
        steps = len(traj.get("steps", []))
        stats["total_steps"] += steps

        # Cost
        cost = traj.get("cost", 0.0)
        stats["total_cost"] += cost

        # Tokens
        tokens = traj.get("total_tokens", 0)
        stats["total_tokens"] += tokens

        # Time
        time = traj.get("total_time", 0.0)
        stats["total_time"] += time

    # Averages
    n = stats["total_instances"]
    if n > 0:
        stats["avg_steps"] = stats["total_steps"] / n
        stats["avg_cost"] = stats["total_cost"] / n
        stats["avg_tokens"] = stats["total_tokens"] / n
        stats["avg_time"] = stats["total_time"] / n
        stats["resolve_rate"] = stats["resolved"] / n

    return stats


def compare_experiments(baseline_stats: dict, memory_stats: dict) -> dict:
    """Compare baseline vs memory statistics.

    Args:
        baseline_stats: Baseline statistics
        memory_stats: Memory statistics

    Returns:
        Dict with comparison
    """
    comparison = {}

    # Resolution rate
    baseline_resolve = baseline_stats.get("resolve_rate", 0.0)
    memory_resolve = memory_stats.get("resolve_rate", 0.0)
    comparison["resolve_rate"] = {
        "baseline": baseline_resolve,
        "memory": memory_resolve,
        "improvement": memory_resolve - baseline_resolve,
        "improvement_pct": ((memory_resolve - baseline_resolve) / baseline_resolve * 100)
        if baseline_resolve > 0
        else 0.0,
    }

    # Cost
    baseline_cost = baseline_stats.get("avg_cost", 0.0)
    memory_cost = memory_stats.get("avg_cost", 0.0)
    comparison["cost"] = {
        "baseline": baseline_cost,
        "memory": memory_cost,
        "difference": memory_cost - baseline_cost,
        "difference_pct": ((memory_cost - baseline_cost) / baseline_cost * 100) if baseline_cost > 0 else 0.0,
    }

    # Tokens
    baseline_tokens = baseline_stats.get("avg_tokens", 0)
    memory_tokens = memory_stats.get("avg_tokens", 0)
    comparison["tokens"] = {
        "baseline": baseline_tokens,
        "memory": memory_tokens,
        "difference": memory_tokens - baseline_tokens,
        "difference_pct": ((memory_tokens - baseline_tokens) / baseline_tokens * 100) if baseline_tokens > 0 else 0.0,
    }

    # Steps
    baseline_steps = baseline_stats.get("avg_steps", 0.0)
    memory_steps = memory_stats.get("avg_steps", 0.0)
    comparison["steps"] = {
        "baseline": baseline_steps,
        "memory": memory_steps,
        "difference": memory_steps - baseline_steps,
        "difference_pct": ((memory_steps - baseline_steps) / baseline_steps * 100) if baseline_steps > 0 else 0.0,
    }

    # Time
    baseline_time = baseline_stats.get("avg_time", 0.0)
    memory_time = memory_stats.get("avg_time", 0.0)
    comparison["time"] = {
        "baseline": baseline_time,
        "memory": memory_time,
        "difference": memory_time - baseline_time,
        "difference_pct": ((memory_time - baseline_time) / baseline_time * 100) if baseline_time > 0 else 0.0,
    }

    return comparison


def print_comparison(comparison: dict):
    """Print comparison in readable format.

    Args:
        comparison: Comparison dict
    """
    print("\n" + "=" * 60)
    print("EXPERIMENT COMPARISON: Baseline vs Memory")
    print("=" * 60)

    print("\n📊 RESOLUTION RATE:")
    print(f"  Baseline:    {comparison['resolve_rate']['baseline']:.1%}")
    print(f"  Memory:      {comparison['resolve_rate']['memory']:.1%}")
    print(
        f"  Improvement: {comparison['resolve_rate']['improvement']:+.1%} "
        f"({comparison['resolve_rate']['improvement_pct']:+.1f}%)"
    )

    print("\n💰 COST (USD):")
    print(f"  Baseline:   ${comparison['cost']['baseline']:.4f}")
    print(f"  Memory:     ${comparison['cost']['memory']:.4f}")
    print(f"  Difference: ${comparison['cost']['difference']:+.4f} ({comparison['cost']['difference_pct']:+.1f}%)")

    print("\n🔢 TOKENS:")
    print(f"  Baseline:   {comparison['tokens']['baseline']:.0f}")
    print(f"  Memory:     {comparison['tokens']['memory']:.0f}")
    print(f"  Difference: {comparison['tokens']['difference']:+.0f} ({comparison['tokens']['difference_pct']:+.1f}%)")

    print("\n👣 STEPS:")
    print(f"  Baseline:   {comparison['steps']['baseline']:.1f}")
    print(f"  Memory:     {comparison['steps']['memory']:.1f}")
    print(f"  Difference: {comparison['steps']['difference']:+.1f} ({comparison['steps']['difference_pct']:+.1f}%)")

    print("\n⏱️  TIME (seconds):")
    print(f"  Baseline:   {comparison['time']['baseline']:.1f}s")
    print(f"  Memory:     {comparison['time']['memory']:.1f}s")
    print(f"  Difference: {comparison['time']['difference']:+.1f}s ({comparison['time']['difference_pct']:+.1f}%)")

    print("\n" + "=" * 60)

    # Summary
    print("\n📝 SUMMARY:")
    if comparison["resolve_rate"]["improvement"] > 0:
        print(f"  ✅ Memory agent resolves {comparison['resolve_rate']['improvement']:.1%} more instances")
    elif comparison["resolve_rate"]["improvement"] < 0:
        print(f"  ❌ Memory agent resolves {abs(comparison['resolve_rate']['improvement']):.1%} fewer instances")
    else:
        print("  ➖ Same resolution rate")

    if comparison["cost"]["difference"] < 0:
        print(f"  ✅ Memory agent saves ${abs(comparison['cost']['difference']):.4f} per instance")
    elif comparison["cost"]["difference"] > 0:
        print(f"  ⚠️  Memory agent costs ${comparison['cost']['difference']:.4f} more per instance")

    if comparison["tokens"]["difference"] < 0:
        print(f"  ✅ Memory agent uses {abs(comparison['tokens']['difference']):.0f} fewer tokens")
    elif comparison["tokens"]["difference"] > 0:
        print(f"  ⚠️  Memory agent uses {comparison['tokens']['difference']:.0f} more tokens")

    print("=" * 60 + "\n")


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Compare baseline vs memory experiment results")
    parser.add_argument(
        "--baseline-dir", type=Path, default=Path("experiments/results/baseline"), help="Baseline results directory"
    )
    parser.add_argument(
        "--memory-dir", type=Path, default=Path("experiments/results/memory"), help="Memory results directory"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("experiments/results/comparison.json"),
        help="Output file for comparison results",
    )

    args = parser.parse_args()

    print("Loading results...")

    # Load results
    baseline_results = load_results(args.baseline_dir)
    memory_results = load_results(args.memory_dir)

    print(f"  Baseline: {len(baseline_results['trajectories'])} instances")
    print(f"  Memory:   {len(memory_results['trajectories'])} instances")

    # Compute statistics
    print("\nComputing statistics...")
    baseline_stats = compute_statistics(baseline_results)
    memory_stats = compute_statistics(memory_results)

    # Compare
    comparison = compare_experiments(baseline_stats, memory_stats)

    # Print comparison
    print_comparison(comparison)

    # Save to file
    output_data = {"baseline_stats": baseline_stats, "memory_stats": memory_stats, "comparison": comparison}

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(output_data, f, indent=2)

    print(f"Comparison saved to: {args.output}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Test A-mem integration with mini-swe-agent on 5 SWE-bench instances."""

import json
import logging
import sys
import time
from pathlib import Path

from datasets import load_dataset

from minisweagent.agents.default import DefaultAgent
from minisweagent.environments.local import LocalEnvironment
from minisweagent.models import get_model

# Setup detailed logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout), logging.FileHandler("experiments/test_detailed.log")],
)
logger = logging.getLogger(__name__)

# Configuration
NUM_INSTANCES = 5
MODEL_NAME = "openai/gpt-4o"  # Use gpt-4o with provider prefix
RESULTS_DIR = Path("experiments/results")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


class VerboseAgent(DefaultAgent):
    """Agent with verbose logging."""

    def step(self) -> dict:
        """Override step to add logging."""
        logger.info(f"🔄 Step {self.model.n_calls + 1} starting...")
        logger.info(f"💰 Current cost: ${self.model.cost:.4f}")
        result = super().step()
        logger.info(f"✅ Step {self.model.n_calls} completed")
        return result

    def query(self) -> dict:
        """Override query to add logging."""
        logger.info(f"🤖 Querying LLM (call #{self.model.n_calls + 1})...")
        logger.info(f"📝 Message history length: {len(self.messages)}")
        result = super().query()
        logger.info("📨 LLM response received")
        return result


def load_test_instances(n=5):
    """Load n instances from SWE-bench Lite."""
    logger.info(f"📥 Loading {n} instances from SWE-bench Lite...")
    dataset = load_dataset("princeton-nlp/SWE-Bench_Lite", split="test")
    instances = [dataset[i] for i in range(min(n, len(dataset)))]
    logger.info(f"✅ Loaded {len(instances)} instances")
    for i, inst in enumerate(instances):
        logger.info(f"  {i + 1}. {inst['instance_id']}")
    return instances


def run_agent_on_instance(agent_class, instance, agent_name, max_steps=10, **agent_kwargs):
    """Run an agent on a single SWE-bench instance with limited steps."""
    instance_id = instance["instance_id"]
    logger.info(f"\n{'=' * 70}")
    logger.info(f"  Running {agent_name} on {instance_id}")
    logger.info(f"{'=' * 70}")

    # Create model
    model_config = {
        "model_name": MODEL_NAME,
        "model_kwargs": {
            "temperature": 0.0,
            "max_tokens": 4096,
        },
    }
    logger.info(f"🔧 Creating model: {MODEL_NAME}")
    model = get_model(MODEL_NAME, model_config)

    # Create environment (local for simplicity)
    logger.info("🏗️  Creating local environment")
    env = LocalEnvironment()

    # Create agent with limited steps for testing
    agent_config = {
        "step_limit": max_steps,  # Limit steps for quick testing
        "cost_limit": 5.0,
    }
    agent_config.update(agent_kwargs)
    logger.info(f"🤖 Creating agent: {agent_class.__name__}")
    logger.info(f"   Step limit: {agent_config['step_limit']}")
    logger.info(f"   Cost limit: ${agent_config['cost_limit']}")

    agent = agent_class(model, env, **agent_config)

    # Create task description
    task = f"""Solve the following GitHub issue:

Repository: {instance["repo"]}
Issue: {instance.get("problem_statement", "No description available")[:500]}...

Please investigate the issue and propose a fix."""

    logger.info(f"📋 Task created (length: {len(task)} chars)")

    # Run agent
    logger.info("🚀 Starting agent execution...")
    start_time = time.time()

    try:
        exit_status, result = agent.run(task)
        success = exit_status == "Finished"
        logger.info(f"✅ Agent finished with status: {exit_status}")
    except KeyboardInterrupt:
        logger.warning("⚠️  Interrupted by user")
        raise
    except Exception as e:
        logger.error(f"❌ Error: {type(e).__name__}: {e}")
        exit_status = type(e).__name__
        result = str(e)
        success = False

    end_time = time.time()
    elapsed = end_time - start_time

    # Collect stats
    stats = {
        "instance_id": instance_id,
        "agent": agent_name,
        "model": MODEL_NAME,
        "success": success,
        "exit_status": exit_status,
        "steps": model.n_calls,
        "cost": model.cost,
        "time": elapsed,
        "result": result[:500] if result else None,  # Truncate
    }

    logger.info(f"\n📊 Results for {instance_id}:")
    logger.info(f"  Status: {exit_status}")
    logger.info(f"  Steps: {stats['steps']}")
    logger.info(f"  Cost: ${stats['cost']:.4f}")
    logger.info(f"  Time: {elapsed:.1f}s")

    return stats


def run_baseline_experiment(instances):
    """Run baseline experiment with DefaultAgent."""
    logger.info("\n" + "=" * 70)
    logger.info("  BASELINE EXPERIMENT (VerboseAgent)")
    logger.info("=" * 70)

    results = []
    for i, instance in enumerate(instances, 1):
        logger.info(f"\n[{i}/{len(instances)}] Processing {instance['instance_id']}")
        stats = run_agent_on_instance(
            VerboseAgent,
            instance,
            "baseline",
            max_steps=10,  # Limited for testing
        )
        results.append(stats)

        # Show progress
        completed = sum(1 for r in results if r["success"])
        logger.info(f"Progress: {i}/{len(instances)} completed, {completed} successful")

    # Save results
    baseline_dir = RESULTS_DIR / "baseline"
    baseline_dir.mkdir(parents=True, exist_ok=True)
    with open(baseline_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)

    logger.info("\n✅ Baseline experiment complete!")
    logger.info(f"   Results saved to: {baseline_dir / 'results.json'}")

    return results


def run_memory_experiment(instances):
    """Run memory experiment with MemoryAgent."""
    logger.info("\n" + "=" * 70)
    logger.info("  MEMORY EXPERIMENT (MemoryAgent + A-mem)")
    logger.info("=" * 70)

    # Note: MemoryAgent doesn't exist yet in mini-swe-agent
    # For now, just run with VerboseAgent
    logger.info("⚠️  Note: Using VerboseAgent for memory experiment (MemoryAgent integration pending)")

    results = []
    for i, instance in enumerate(instances, 1):
        logger.info(f"\n[{i}/{len(instances)}] Processing {instance['instance_id']}")
        stats = run_agent_on_instance(
            VerboseAgent,
            instance,
            "memory",
            max_steps=10,  # Limited for testing
        )
        results.append(stats)

        # Show progress
        completed = sum(1 for r in results if r["success"])
        logger.info(f"Progress: {i}/{len(instances)} completed, {completed} successful")

    # Save results
    memory_dir = RESULTS_DIR / "memory"
    memory_dir.mkdir(parents=True, exist_ok=True)
    with open(memory_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)

    logger.info("\n✅ Memory experiment complete!")
    logger.info(f"   Results saved to: {memory_dir / 'results.json'}")

    return results


def compare_results(baseline_results, memory_results):
    """Compare baseline vs memory results."""
    logger.info("\n" + "=" * 70)
    logger.info("  COMPARISON: Baseline vs Memory")
    logger.info("=" * 70)

    baseline_success = sum(1 for r in baseline_results if r["success"])
    memory_success = sum(1 for r in memory_results if r["success"])

    baseline_cost = sum(r["cost"] for r in baseline_results)
    memory_cost = sum(r["cost"] for r in memory_results)

    baseline_steps = sum(r["steps"] for r in baseline_results)
    memory_steps = sum(r["steps"] for r in memory_results)

    baseline_time = sum(r["time"] for r in baseline_results)
    memory_time = sum(r["time"] for r in memory_results)

    n = len(baseline_results)

    logger.info("\n📊 Success Rate:")
    logger.info(f"  Baseline: {baseline_success}/{n} ({baseline_success / n * 100:.1f}%)")
    logger.info(f"  Memory:   {memory_success}/{n} ({memory_success / n * 100:.1f}%)")

    logger.info("\n💰 Total Cost:")
    logger.info(f"  Baseline: ${baseline_cost:.4f}")
    logger.info(f"  Memory:   ${memory_cost:.4f}")
    logger.info(f"  Diff:     ${memory_cost - baseline_cost:+.4f}")

    logger.info("\n👣 Average Steps:")
    logger.info(f"  Baseline: {baseline_steps / n:.1f}")
    logger.info(f"  Memory:   {memory_steps / n:.1f}")
    logger.info(f"  Diff:     {(memory_steps - baseline_steps) / n:+.1f}")

    logger.info("\n⏱️  Total Time:")
    logger.info(f"  Baseline: {baseline_time:.1f}s")
    logger.info(f"  Memory:   {memory_time:.1f}s")
    logger.info(f"  Diff:     {memory_time - baseline_time:+.1f}s")

    # Save comparison
    comparison = {
        "baseline": {
            "success_rate": baseline_success / n,
            "total_cost": baseline_cost,
            "avg_steps": baseline_steps / n,
            "total_time": baseline_time,
        },
        "memory": {
            "success_rate": memory_success / n,
            "total_cost": memory_cost,
            "avg_steps": memory_steps / n,
            "total_time": memory_time,
        },
        "diff": {
            "success_rate": (memory_success - baseline_success) / n,
            "cost": memory_cost - baseline_cost,
            "steps": (memory_steps - baseline_steps) / n,
            "time": memory_time - baseline_time,
        },
    }

    with open(RESULTS_DIR / "comparison.json", "w") as f:
        json.dump(comparison, f, indent=2)

    logger.info(f"\n✅ Comparison saved to: {RESULTS_DIR / 'comparison.json'}")


def main():
    """Main function."""
    logger.info("=" * 70)
    logger.info("  A-mem Integration Test for Mini-SWE-Agent")
    logger.info(f"  Testing on {NUM_INSTANCES} SWE-bench Lite instances")
    logger.info(f"  Model: {MODEL_NAME}")
    logger.info("=" * 70)

    # Load instances
    instances = load_test_instances(NUM_INSTANCES)

    # Run baseline
    baseline_results = run_baseline_experiment(instances)

    # Run memory
    memory_results = run_memory_experiment(instances)

    # Compare
    compare_results(baseline_results, memory_results)

    logger.info("\n" + "=" * 70)
    logger.info("  🎉 All experiments complete!")
    logger.info("=" * 70)
    logger.info(f"\nResults saved in: {RESULTS_DIR}")
    logger.info("Detailed log: experiments/test_detailed.log")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.warning("\n⚠️  Experiment interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"\n❌ Fatal error: {e}", exc_info=True)
        sys.exit(1)

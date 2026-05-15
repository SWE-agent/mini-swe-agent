"""Ablation runner: toggle each PlanMem component and compare on a small subset.

Each variant is a `PlanMemConfig` with a different combination of feature
flags. The goal is to demonstrate that planner / adaptive memory / replan /
memory→planner each carry their own weight, not just the combination.

Usage:
    python -m experiments.run_ablation --instances 5 --model deepseek/deepseek-chat

Writes one preds JSON per variant under outputs/ablation/<variant>/preds.json
plus a summary table.
"""

from __future__ import annotations

import argparse
import json
import logging
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Variant:
    """One row in the ablation table.

    The flags map onto ``PlanMemConfig`` fields. ``baseline`` disables both
    the planner and adaptive memory, so it falls back to the parent
    ``MemorySearchAgent``'s fixed retrieval — useful as a control.
    """

    name: str
    flags: dict = field(default_factory=dict)


VARIANTS: list[Variant] = [
    Variant(
        "baseline",
        {
            "enable_planner": False,
            "enable_adaptive_memory": False,
            "enable_replanning": False,
            "enable_memory_to_planner": False,
            "use_llm_decomposition": False,
            "use_llm_replan": False,
        },
    ),
    Variant(
        "planner_only",
        {
            "enable_planner": True,
            "enable_adaptive_memory": False,
            "enable_replanning": False,
            "enable_memory_to_planner": False,
            "use_llm_decomposition": False,
            "use_llm_replan": False,
        },
    ),
    Variant(
        "adaptive_only",
        {
            "enable_planner": True,  # adaptive needs planner-derived signal
            "enable_adaptive_memory": True,
            "enable_replanning": False,
            "enable_memory_to_planner": False,
            "use_llm_decomposition": False,
            "use_llm_replan": False,
        },
    ),
    Variant(
        "plus_mem_to_planner",
        {
            "enable_planner": True,
            "enable_adaptive_memory": True,
            "enable_replanning": False,
            "enable_memory_to_planner": True,
            "use_llm_decomposition": False,
            "use_llm_replan": False,
        },
    ),
    Variant(
        "full",
        {
            "enable_planner": True,
            "enable_adaptive_memory": True,
            "enable_replanning": True,
            "enable_memory_to_planner": True,
            "use_llm_decomposition": True,
            "use_llm_replan": False,  # keep recovery deterministic for fair compare
        },
    ),
]


def run_one_variant(
    variant: Variant,
    instances: list,
    *,
    model_name: str,
    base_config: dict,
    out_root: Path,
) -> dict:
    """Run all instances under one variant. Returns summary dict."""
    from minisweagent.run.extra.swebench import get_sb_environment
    from minisweagent.run.utils.save import save_traj

    from minisweagent.agents.planmem_agent import PlanMemAgent
    from minisweagent.models import get_model

    out_dir = out_root / variant.name
    out_dir.mkdir(parents=True, exist_ok=True)
    preds: dict = {}
    durations: list[float] = []
    errors: int = 0

    for i, instance in enumerate(instances):
        instance_id = instance["instance_id"]
        traj_path = out_dir / f"{instance_id}.traj.json"
        if traj_path.exists():
            continue
        try:
            cfg = dict(base_config)
            cfg.setdefault("environment", {})["environment_class"] = "docker"
            env = get_sb_environment(cfg, instance)
            model = get_model(model_name, cfg.get("model", {}))

            agent_kwargs = dict(cfg.get("agent", {}))
            agent_kwargs.update(variant.flags)
            agent = PlanMemAgent(model=model, env=env, **agent_kwargs)

            t0 = time.perf_counter()
            exit_status, result = agent.run(instance["problem_statement"])
            durations.append(time.perf_counter() - t0)

            save_traj(agent, traj_path, exit_status=exit_status, result=result)
            patch = ""
            try:
                with open(traj_path) as f:
                    data = json.load(f)
                patch = data.get("info", {}).get("submission", "") or ""
            except Exception:
                pass
            if patch:
                preds[instance_id] = {
                    "model_patch": patch,
                    "model_name_or_path": f"planmem-{variant.name}",
                }
            print(f"[{variant.name}] [{i + 1}/{len(instances)}] {instance_id}: {exit_status}")
        except Exception as e:
            errors += 1
            print(f"[{variant.name}] [{i + 1}/{len(instances)}] {instance_id}: ERROR {e}")
            logger.warning("instance failed", exc_info=True)

    preds_path = out_dir / "preds.json"
    preds_path.write_text(json.dumps(preds, indent=2))

    return {
        "variant": variant.name,
        "completed": len(preds),
        "errors": errors,
        "mean_seconds": (sum(durations) / len(durations)) if durations else 0.0,
        "preds_path": str(preds_path),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--instances", type=int, default=3, help="Number of SWE-bench instances per variant")
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--config", type=str, default=None, help="Optional path to a YAML agent config")
    parser.add_argument("--out", type=str, default="experiments/outputs/ablation")
    parser.add_argument("--variants", type=str, default=None, help="Comma-separated subset of variant names to run")
    args = parser.parse_args()

    import yaml
    from datasets import load_dataset

    from minisweagent.config import builtin_config_dir

    config_path = Path(args.config) if args.config else (builtin_config_dir / "extra" / "swebench.yaml")
    base_config = yaml.safe_load(Path(config_path).read_text())

    print(f"Loading SWE-bench Verified, taking first {args.instances} instances...")
    dataset = load_dataset("princeton-nlp/SWE-bench_Verified", split="test")
    instances = list(dataset)[: args.instances]

    selected = VARIANTS
    if args.variants:
        wanted = {n.strip() for n in args.variants.split(",") if n.strip()}
        selected = [v for v in VARIANTS if v.name in wanted]

    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)

    summaries = []
    for variant in selected:
        print(f"\n=== Variant: {variant.name} ===")
        try:
            summaries.append(
                run_one_variant(
                    variant,
                    instances,
                    model_name=args.model,
                    base_config=base_config,
                    out_root=out_root,
                )
            )
        except Exception:
            traceback.print_exc()

    # Summary table
    print()
    header = f"{'variant':<22} {'completed':>10} {'errors':>7} {'mean_s':>9}"
    print(header)
    print("-" * len(header))
    for s in summaries:
        print(f"{s['variant']:<22} {s['completed']:>10} {s['errors']:>7} {s['mean_seconds']:>9.1f}")

    summary_path = out_root / "summary.json"
    summary_path.write_text(json.dumps(summaries, indent=2))
    print(f"\nSummary written to {summary_path}")


if __name__ == "__main__":
    main()

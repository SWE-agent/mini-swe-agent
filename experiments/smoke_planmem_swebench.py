"""Single-instance PlanMem smoke test on SWE-bench Verified.

Use this BEFORE the batch runner to confirm:
- vLLM is reachable and the model is loaded
- Docker image for the chosen instance pulls / runs
- PlanMemAgent's run loop terminates with a Submitted exit_status

Run:
    # First start vLLM (see docstring at the top of swebench_planmem.yaml).
    python -m experiments.smoke_planmem_swebench \\
        --instance astropy__astropy-12907 \\
        --model openai/Qwen/Qwen3-Coder-30B-A3B-Instruct-FP8 \\
        --api-base http://localhost:8765/v1
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import yaml

CONFIG_PATH = Path("src/minisweagent/config/benchmarks/swebench_planmem.yaml")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--instance", default="astropy__astropy-12907",
                        help="SWE-bench Verified instance_id")
    parser.add_argument("--model", default="openai/Qwen/Qwen3-Coder-30B-A3B-Instruct-FP8")
    parser.add_argument("--api-base", default="http://localhost:8765/v1")
    parser.add_argument("--config", default=str(CONFIG_PATH))
    parser.add_argument("--out", default="experiments/outputs/planmem_smoke")
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg = yaml.safe_load(Path(args.config).read_text())
    cfg.setdefault("model", {})["model_name"] = args.model
    mk = cfg["model"].setdefault("model_kwargs", {})
    mk["api_base"] = args.api_base
    mk.setdefault("api_key", "EMPTY")
    # api_base is NOT a top-level model field; clean up if it was put there.
    cfg["model"].pop("api_base", None)
    cfg.setdefault("environment", {})["environment_class"] = "docker"

    from datasets import load_dataset

    from minisweagent.agents.planmem_agent import PlanMemAgent
    from minisweagent.models import get_model
    from minisweagent.run.benchmarks.swebench import get_sb_environment

    print(f"Loading SWE-bench Verified, picking instance {args.instance!r}...")
    ds = load_dataset("princeton-nlp/SWE-Bench_Verified", split="test")
    matches = [i for i in ds if i["instance_id"] == args.instance]
    if not matches:
        print(f"ERROR: instance not found in dataset", file=sys.stderr)
        return 2
    instance = matches[0]

    print(f"Pulling docker image / starting environment for {args.instance}...")
    env = get_sb_environment(cfg, instance)
    model = get_model(config=cfg.get("model", {}))

    agent_kwargs = dict(cfg.get("agent", {}))
    agent_kwargs.pop("agent_class", None)
    agent = PlanMemAgent(model, env, **agent_kwargs)

    print(f"Planner status: {agent.planner.progress if agent.config.enable_planner else 'disabled'}")
    print(f"Running PlanMemAgent for up to {agent.config.step_limit} steps "
          f"or ${agent.config.cost_limit:.2f}...")
    t0 = time.perf_counter()
    info = agent.run(instance["problem_statement"])
    dt = time.perf_counter() - t0

    traj_path = out_dir / f"{args.instance}.traj.json"
    agent.save(traj_path, {
        "info": info,
        "instance_id": args.instance,
        "planner_progress": agent.planner.progress
            if agent.config.enable_planner else "disabled",
    })

    print()
    print(f"exit_status   : {info.get('exit_status')}")
    print(f"submission len: {len(info.get('submission', '') or '')}")
    print(f"n_calls       : {agent.n_calls}")
    print(f"cost          : ${agent.cost:.4f}")
    print(f"wall time     : {dt:.1f}s")
    print(f"planner       : {agent.planner.progress if agent.config.enable_planner else 'disabled'}")
    print(f"trajectory    : {traj_path}")

    # Quick metadata sanity check (codex flagged this as a likely break in v2).
    has_filenames = sum(
        1 for n in agent.memory_graph
        if isinstance(n.metadata, dict) and n.metadata.get("filenames")
    )
    print(f"nodes w/ filename metadata: {has_filenames} (>0 = action-extraction OK)")

    # Exit code = 0 only on Submitted, so a CI loop can rely on it.
    return 0 if info.get("exit_status") == "Submitted" else 1


if __name__ == "__main__":
    sys.exit(main())

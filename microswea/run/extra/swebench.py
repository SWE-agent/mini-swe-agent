#!/usr/bin/env python3
import json
from pathlib import Path
from typing import Literal

import typer
import yaml
from datasets import load_dataset

from microswea import package_dir
from microswea.agents.micro import Agent
from microswea.environments.docker import DockerEnvironment
from microswea.models.litellm_model import LitellmModel

app = typer.Typer()

DATASET_MAPPING = {
    "full": "princeton-nlp/SWE-Bench",
    "verified": "princeton-nlp/SWE-Bench_Verified",
    "lite": "princeton-nlp/SWE-Bench_Lite",
    "multimodal": "princeton-nlp/SWE-Bench_Multimodal",
    "multilingual": "swe-bench/SWE-Bench_Multilingual",
    "_test": "klieret/swe-bench-dummy-test-dataset",
}


def get_image_name(instance: dict) -> str:
    """Get the image name for a SWEBench instance."""
    image_name = instance.get("image_name", None)
    if image_name is None:
        # Docker doesn't allow double underscore, so we replace them with a magic token
        iid = instance["instance_id"]
        id_docker_compatible = iid.replace("__", "_1776_")
        image_name = f"swebench/sweb.eval.x86_64.{id_docker_compatible}:latest".lower()
    return image_name


def update_output_file(output_path: Path, instance_id: str, model_config, result: str):
    """Update the output JSON file with results from a single instance."""
    output_data = {}
    if output_path.exists():
        output_data = json.loads(output_path.read_text())

    output_data[instance_id] = {
        "model_name_or_path": model_config.model_name,
        "instance_id": instance_id,
        "model_patch": result,
    }

    output_path.write_text(json.dumps(output_data, indent=2))


def process_instance(instance: dict, output_path: Path) -> dict:
    """Process a single SWEBench instance."""
    instance_id = instance["instance_id"]
    problem_statement = instance["problem_statement"]

    config = yaml.safe_load((package_dir / "config" / "extra" / "swebench.yaml").read_text())
    image_name = get_image_name(instance)

    agent = Agent(
        LitellmModel(**config["model"]),
        DockerEnvironment(**(config["environment"] | {"image": image_name})),
        problem_statement,
        **config["agent"],
    )

    try:
        result = agent.run()
    finally:
        Path("traj.json").write_text(
            json.dumps(agent.history, indent=2),
        )

    update_output_file(output_path, instance_id, agent.model.config, result)

    print(f"Instance {instance_id} completed")
    print(f"Cost: ${agent.model.cost:.4f}")
    print(f"Steps: {agent.model.n_calls}")

    return {
        "instance_id": instance_id,
        "result": result,
        "cost": agent.model.cost,
        "steps": agent.model.n_calls,
    }


def process_instances(instances: list[dict], output_path: Path):
    """Process a list of SWEBench instances."""
    results = []
    running_cost = 0.0

    for i, instance in enumerate(instances):
        instance_id = instance["instance_id"]

        print(f"\n{'=' * 60}")
        print(f"Running instance {i + 1}/{len(instances)}: {instance_id}")
        print(f"{'=' * 60}")

        result = process_instance(instance, output_path)
        results.append(result)
        running_cost += result["cost"]

        # Print summary after each instance
        print(f"\nRunning total - Instances: {i + 1}/{len(instances)}, Cost: ${running_cost:.4f}")


def run_swebench(subset: str = "lite", split: str = "dev", slice_spec: str = "", output: str = "results.json"):
    """Run micro-SWE-agent on SWEBench instances.

    Args:
        subset: SWEBench subset to use ("lite", "verified", "full", "multimodal", "multilingual")
        split: Dataset split ("dev" or "test")
        slice_spec: Slice specification (e.g., "0:5" for first 5 instances)
        output: Output JSON file path
    """
    dataset_path = DATASET_MAPPING[subset]
    print(f"Loading dataset {dataset_path}, split {split}...")
    instances = list(load_dataset(dataset_path, split=split))

    if slice_spec:
        values = [int(x) if x else None for x in slice_spec.split(":")]
        instances = instances[slice(*values)]

    output_path = Path(output)
    print(f"Running on {len(instances)} instances...")
    print(f"Results will be saved to {output_path}")

    process_instances(instances, output_path)


@app.command()
def main(
    subset: Literal["full", "verified", "lite", "multimodal", "multilingual", "_test"] = typer.Option(
        "lite", "--subset", help="SWEBench subset to use"
    ),
    split: Literal["dev", "test"] = typer.Option("dev", "--split", help="Dataset split"),
    slice_spec: str = typer.Option("", "--slice", help="Slice specification (e.g., '0:5' for first 5 instances)"),
    output: str = typer.Option("results.json", "-o", "--output", help="Output JSON file path"),
) -> None:
    """Run micro-SWE-agent on SWEBench instances"""
    run_swebench(subset=subset, split=split, slice_spec=slice_spec, output=output)


if __name__ == "__main__":
    app()

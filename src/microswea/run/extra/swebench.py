#!/usr/bin/env python3
import concurrent.futures
import contextlib
import io
import json
import random
import re
from pathlib import Path
from typing import Literal

import typer
import yaml
from datasets import load_dataset

from microswea import package_dir
from microswea.agents.default import DefaultAgent
from microswea.environments.docker import DockerEnvironment
from microswea.models import get_model

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

    agent = DefaultAgent(
        get_model(config=config.get("model", {})),
        DockerEnvironment(**(config.get("environment", {}) | {"image": image_name})),
        problem_statement,
        **config.get("agent", {}),
    )

    try:
        result = agent.run()
    finally:
        Path("traj.json").write_text(
            json.dumps(agent.messages, indent=2),
        )

    update_output_file(output_path, instance_id, agent.model.config, result)

    return {
        "instance_id": instance_id,
        "result": result,
        "cost": agent.model.cost,
        "steps": agent.model.n_calls,
    }


def process_instances_single_threaded(instances: list[dict], output_path: Path):
    """Process SWEBench instances sequentially."""
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

        print(f"\nRunning total - Instances: {i + 1}/{len(instances)}, Cost: ${running_cost:.4f}")

    return results


def process_instances_multithreaded(instances: list[dict], output_path: Path, n_workers: int):
    """Process SWEBench instances in parallel."""
    results = []
    running_cost = 0.0

    def process_with_captured_output(instance):
        instance_id = instance["instance_id"]

        print(f"Starting instance {instance_id}")
        with contextlib.redirect_stdout(io.StringIO()):
            result = process_instance(instance, output_path)
        print(f"Instance {instance_id} completed (${result['cost']:.4f}, {result['steps']} steps)")

        return result

    print(f"Starting parallel execution with {n_workers} workers...")

    with concurrent.futures.ThreadPoolExecutor(max_workers=n_workers) as executor:
        futures = [executor.submit(process_with_captured_output, instance) for instance in instances]

        for future in concurrent.futures.as_completed(futures):
            result = future.result()
            # result["instance_id"] already has the ID we need
            results.append(result)
            running_cost += result["cost"]

            completed = len(results)
            print(f"\nRunning total - Instances: {completed}/{len(instances)}, Cost: ${running_cost:.4f}")

    return results


def process_instances(instances: list[dict], output_path: Path, n_workers: int = 1):
    """Process a list of SWEBench instances."""
    if n_workers == 1:
        return process_instances_single_threaded(instances, output_path)
    return process_instances_multithreaded(instances, output_path, n_workers)


def filter_instances(
    instances: list[dict], *, filter_spec: str, slice_spec: str = "", shuffle: bool = False
) -> list[dict]:
    """Filter and slice a list of SWEBench instances."""
    if shuffle:
        instances = sorted(instances.copy(), key=lambda x: x["instance_id"])
        random.seed(42)
        random.shuffle(instances)
    before_filter = len(instances)
    instances = [instance for instance in instances if re.match(filter_spec, instance["instance_id"])]
    if (after_filter := len(instances)) != before_filter:
        print(f"Instance filter: {before_filter} -> {after_filter} instances")
    if slice_spec:
        values = [int(x) if x else None for x in slice_spec.split(":")]
        instances = instances[slice(*values)]
        if (after_slice := len(instances)) != before_filter:
            print(f"Instance slice: {before_filter} -> {after_slice} instances")
    return instances


@app.command()
def main(
    subset: str = typer.Option("lite", "--subset", help="SWEBench subset to use or path to a dataset"),
    split: Literal["dev", "test"] = typer.Option("dev", "--split", help="Dataset split"),
    slice_spec: str = typer.Option("", "--slice", help="Slice specification (e.g., '0:5' for first 5 instances)"),
    filter_spec: str = typer.Option("", "--filter", help="Filter instance IDs by regex"),
    shuffle: bool = typer.Option(False, "--shuffle", help="Shuffle instances"),
    output: str = typer.Option("results.json", "-o", "--output", help="Output JSON file path"),
    n_workers: int = typer.Option(1, "--n-workers", help="Number of worker threads for parallel processing"),
) -> None:
    """Run micro-SWE-agent on SWEBench instances"""
    try:
        dataset_path = DATASET_MAPPING[subset]
    except KeyError:
        dataset_path = subset
    print(f"Loading dataset {dataset_path}, split {split}...")
    instances = list(load_dataset(dataset_path, split=split))

    instances = filter_instances(instances, filter_spec=filter_spec, slice_spec=slice_spec, shuffle=shuffle)

    output_path = Path(output)
    print(f"Running on {len(instances)} instances...")
    print(f"Results will be saved to {output_path}")

    process_instances(instances, output_path, n_workers)


if __name__ == "__main__":
    app()

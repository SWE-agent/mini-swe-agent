#!/usr/bin/env python3
import concurrent.futures
import json
import random
import re
from pathlib import Path

import typer
import yaml
from datasets import load_dataset
from rich.live import Live

from microswea import package_dir
from microswea.agents.default import DefaultAgent
from microswea.environments.docker import DockerEnvironment
from microswea.models import get_model
from microswea.run.extra.utils.batch_progress import RunBatchProgressManager

app = typer.Typer()

DATASET_MAPPING = {
    "full": "princeton-nlp/SWE-Bench",
    "verified": "princeton-nlp/SWE-Bench_Verified",
    "lite": "princeton-nlp/SWE-Bench_Lite",
    "multimodal": "princeton-nlp/SWE-Bench_Multimodal",
    "multilingual": "swe-bench/SWE-Bench_Multilingual",
    "_test": "klieret/swe-bench-dummy-test-dataset",
}


class ProgressTrackingAgent(DefaultAgent):
    """Simple wrapper around DefaultAgent that provides progress updates."""

    def __init__(self, *args, progress_manager=None, instance_id=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.progress_manager = progress_manager
        self.instance_id = instance_id

    def step(self) -> str:
        """Override step to provide progress updates."""
        if self.progress_manager and self.instance_id:
            self.progress_manager.update_instance_status(
                self.instance_id, f"Step {self.model.n_calls + 1} (${self.model.cost:.3f})"
            )
        return super().step()


def get_swebench_docker_image_name(instance: dict) -> str:
    """Get the image name for a SWEBench instance."""
    image_name = instance.get("image_name", None)
    if image_name is None:
        # Docker doesn't allow double underscore, so we replace them with a magic token
        iid = instance["instance_id"]
        id_docker_compatible = iid.replace("__", "_1776_")
        image_name = f"swebench/sweb.eval.x86_64.{id_docker_compatible}:latest".lower()
    return image_name


def update_preds_file(output_path: Path, instance_id: str, model_config, result: str):
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


def process_instance(instance: dict, output_dir: Path, model: str, progress_manager=None) -> dict:
    """Process a single SWEBench instance."""
    instance_id = instance["instance_id"]
    task = instance["problem_statement"]

    config = yaml.safe_load((package_dir / "config" / "extra" / "swebench.yaml").read_text())
    image_name = get_swebench_docker_image_name(instance)

    agent = ProgressTrackingAgent(
        get_model(model, config=config.get("model", {})),
        DockerEnvironment(**(config.get("environment", {}) | {"image": image_name})),
        progress_manager=progress_manager,
        instance_id=instance_id,
        **config.get("agent", {}),
    )

    try:
        exit_status, result = agent.run(task)
    except Exception as e:
        exit_status, result = type(e).__name__, str(e)

    data = {
        "instance_id": instance_id,
        "info": {
            "exit_status": exit_status,
            "submission": result,
            "model_stats": {
                "instance_cost": agent.model.cost,
                "api_calls": agent.model.n_calls,
            },
        },
        "messages": agent.messages,
    }
    instance_dir = output_dir / instance_id
    instance_dir.mkdir(parents=True, exist_ok=True)
    (instance_dir / f"{instance_id}.traj.json").write_text(json.dumps(data, indent=2))
    update_preds_file(output_dir / "preds.json", instance_id, agent.model.config, result)

    return data


def process_instances_single_threaded(instances: list[dict], output_path: Path, model: str):
    """Process SWEBench instances sequentially."""
    results = []
    running_cost = 0.0

    for i, instance in enumerate(instances):
        instance_id = instance["instance_id"]

        print(f"Running instance {i + 1}/{len(instances)}: {instance_id}")
        result = process_instance(instance, output_path, model)
        running_cost += result["info"]["model_stats"]["instance_cost"]
        print(f"Instance {instance_id} completed - completed {i + 1}/{len(instances)}, ${running_cost:.4f}")
        results.append(result)

    return results


def process_instances_multithreaded(instances: list[dict], output_path: Path, n_workers: int, model: str):
    """Process SWEBench instances in parallel."""
    results = []

    # Create progress manager
    progress_manager = RunBatchProgressManager(len(instances))

    print(f"Starting parallel execution with {n_workers} workers...")

    with Live(progress_manager.render_group, refresh_per_second=4):
        with concurrent.futures.ThreadPoolExecutor(max_workers=n_workers) as executor:
            # Start all instances and track them
            instance_futures = {}
            for instance in instances:
                instance_id = instance["instance_id"]
                progress_manager.on_instance_start(instance_id)
                future = executor.submit(process_instance, instance, output_path, model, progress_manager)
                instance_futures[future] = instance_id

            for future in concurrent.futures.as_completed(instance_futures):
                instance_id = instance_futures[future]
                try:
                    result = future.result()
                    results.append(result)
                    exit_status = result["info"]["exit_status"]
                    progress_manager.on_instance_end(instance_id, exit_status)
                except Exception as e:
                    progress_manager.on_uncaught_exception(instance_id, e)
                    raise

    return results


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
    split: str = typer.Option("dev", "--split", help="Dataset split"),
    slice_spec: str = typer.Option("", "--slice", help="Slice specification (e.g., '0:5' for first 5 instances)"),
    filter_spec: str = typer.Option("", "--filter", help="Filter instance IDs by regex"),
    shuffle: bool = typer.Option(False, "--shuffle", help="Shuffle instances"),
    output: str = typer.Option("", "-o", "--output", help="Output directory"),
    workers: int = typer.Option(1, "--workers", help="Number of worker threads for parallel processing"),
    model: str = typer.Option("", "--model", help="Model to use"),
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
    output_path.mkdir(parents=True, exist_ok=True)
    print(f"Running on {len(instances)} instances...")
    print(f"Results will be saved to {output_path}")

    if workers == 1:
        process_instances_single_threaded(instances, output_path, model)
    else:
        process_instances_multithreaded(instances, output_path, workers, model)


if __name__ == "__main__":
    app()

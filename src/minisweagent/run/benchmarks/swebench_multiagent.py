#!/usr/bin/env python3

"""Run the multi-agent reviewer system on SWE-bench instances in batch mode."""

import concurrent.futures
import copy
import json
import threading
import time
from pathlib import Path

import typer
from rich.live import Live

from minisweagent.agents import get_agent
from minisweagent.agents.multi_agent import MultiAgentOrchestrator
from minisweagent.config import builtin_config_dir, get_config_from_spec
from minisweagent.environments import get_environment
from minisweagent.models import get_model
from minisweagent.run.benchmarks.swebench import (
    DATASET_MAPPING,
    filter_instances,
    get_swebench_docker_image_name,
    remove_from_preds_file,
    update_preds_file,
)
from minisweagent.run.benchmarks.utils.batch_progress import RunBatchProgressManager
from minisweagent.utils.log import add_file_handler, logger
from minisweagent.utils.serialize import recursive_merge

_HELP_TEXT = """Run the multi-agent reviewer system on SWE-bench instances.

A main agent attempts the fix; on clean confident exit a reviewer agent independently
verifies (and optionally fixes) the work. The reviewer's patch is the final submission.
"""

_CONFIG_SPEC_HELP_TEXT = """Path to config files, filenames, or key-value pairs.

[bold red]IMPORTANT:[/bold red] If you set this option, the default config file will not be used.
Add it explicitly: [bold green]-c swebench_multiagent.yaml <other options>[/bold green]
"""

DEFAULT_CONFIG_FILE = builtin_config_dir / "benchmarks" / "swebench_multiagent.yaml"

app = typer.Typer(rich_markup_mode="rich", add_completion=False)
_OUTPUT_FILE_LOCK = threading.Lock()


def process_instance(
    instance: dict,
    output_dir: Path,
    config: dict,
    progress_manager: RunBatchProgressManager,
) -> None:
    """Process a single SWE-bench instance with the multiagent system."""
    instance_id = instance["instance_id"]
    instance_dir = output_dir / instance_id
    remove_from_preds_file(output_dir / "preds.json", instance_id)
    traj_path = instance_dir / f"{instance_id}.traj.json"
    traj_path.unlink(missing_ok=True)

    task = instance["problem_statement"]
    exit_status = None
    result = None
    model_name = "unknown"

    progress_manager.on_instance_start(instance_id)
    progress_manager.update_instance_status(instance_id, "Pulling/starting environment")

    orchestrator = None
    try:
        # Build shared environment with all submission signals registered
        env_config = copy.deepcopy(config.get("environment", {}))
        env_config["environment_class"] = env_config.get("environment_class", "docker")
        image_name = get_swebench_docker_image_name(instance)
        if env_config["environment_class"] in ["docker", "swerex_modal"]:
            env_config["image"] = image_name
        elif env_config["environment_class"] in ["singularity", "contree"]:
            env_config["image"] = "docker://" + image_name
        env = get_environment(env_config)

        # Merge shared model config (observation_template, format_error_template)
        # into each role-specific model config
        shared_model = config.get("model", {})
        main_model_config = recursive_merge(copy.deepcopy(shared_model), config.get("main_model", {}))
        reviewer_model_config = recursive_merge(copy.deepcopy(shared_model), config.get("reviewer_model", {}))
        model_name = reviewer_model_config.get("model_name") or main_model_config.get("model_name") or model_name

        # Build main agent
        main_model = get_model(config=main_model_config)
        main_agent_config = copy.deepcopy(config.get("main_agent", {}))
        main_agent_config.pop("output_path", None)  # orchestrator handles saving
        main_agent = get_agent(main_model, env, main_agent_config, default_type="default")

        def on_step(label: str, n: int, cost: float):
            progress_manager.update_instance_status(instance_id, f"{label} step {n:3d} (${cost:.2f})")

        progress_manager.update_instance_status(instance_id, "Running main agent")

        orchestrator = MultiAgentOrchestrator(
            env=env,
            main_agent=main_agent,
            reviewer_model_config=reviewer_model_config,
            reviewer_agent_config=copy.deepcopy(config.get("reviewer_agent", {})),
            output_path=traj_path,
            on_step=on_step,
            **config.get("orchestrator", {}),
        )
        info = orchestrator.run(task)
        exit_status = info.get("info", {}).get("exit_status")
        result = info.get("info", {}).get("submission", "")

    except Exception as e:
        logger.error(f"Error processing instance {instance_id}: {e}", exc_info=True)
        exit_status, result = type(e).__name__, ""
        if orchestrator is not None:
            orchestrator.save(submission="", exit_status=exit_status)
    finally:
        update_preds_file(output_dir / "preds.json", instance_id, model_name, result or "")
        progress_manager.on_instance_end(instance_id, exit_status)
        if traj_path.exists():
            logger.info(f"Saved trajectory to '{traj_path}'")


# fmt: off
@app.command(help=_HELP_TEXT)
def main(
    subset: str = typer.Option("lite", "--subset", help="SWEBench subset to use or path to a dataset", rich_help_panel="Data selection"),
    split: str = typer.Option("dev", "--split", help="Dataset split", rich_help_panel="Data selection"),
    slice_spec: str = typer.Option("", "--slice", help="Slice specification (e.g., '0:5' for first 5 instances)", rich_help_panel="Data selection"),
    filter_spec: str = typer.Option("", "--filter", help="Filter instance IDs by regex", rich_help_panel="Data selection"),
    shuffle: bool = typer.Option(False, "--shuffle", help="Shuffle instances", rich_help_panel="Data selection"),
    output: str = typer.Option("", "-o", "--output", help="Output directory", rich_help_panel="Basic"),
    workers: int = typer.Option(1, "-w", "--workers", help="Number of worker threads for parallel processing", rich_help_panel="Basic"),
    redo_existing: bool = typer.Option(False, "--redo-existing", help="Redo existing instances", rich_help_panel="Data selection"),
    config_spec: list[str] = typer.Option([str(DEFAULT_CONFIG_FILE)], "-c", "--config", help=_CONFIG_SPEC_HELP_TEXT, rich_help_panel="Basic"),
    max_review_iterations: int | None = typer.Option(None, "--max-review-iterations", help="Override max reviewer passes", rich_help_panel="Advanced"),
) -> None:
    # fmt: on
    output_path = Path(output)
    output_path.mkdir(parents=True, exist_ok=True)
    logger.info(f"Results will be saved to {output_path}")
    add_file_handler(output_path / "minisweagent.log")

    from datasets import load_dataset

    dataset_path = DATASET_MAPPING.get(subset, subset)
    logger.info(f"Loading dataset {dataset_path}, split {split}...")
    instances = list(load_dataset(dataset_path, split=split))

    instances = filter_instances(instances, filter_spec=filter_spec, slice_spec=slice_spec, shuffle=shuffle)
    if not redo_existing and (output_path / "preds.json").exists():
        existing_instances = list(json.loads((output_path / "preds.json").read_text()).keys())
        logger.info(f"Skipping {len(existing_instances)} existing instances")
        instances = [i for i in instances if i["instance_id"] not in existing_instances]
    logger.info(f"Running on {len(instances)} instances...")

    logger.info(f"Building config from specs: {config_spec}")
    configs = [get_config_from_spec(spec) for spec in config_spec]
    if max_review_iterations is not None:
        configs.append({"orchestrator": {"max_review_iterations": max_review_iterations}})
    config = recursive_merge(*configs)

    progress_manager = RunBatchProgressManager(len(instances), output_path / f"exit_statuses_{time.time()}.yaml")

    def process_futures(futures: dict[concurrent.futures.Future, str]):
        for future in concurrent.futures.as_completed(futures):
            try:
                future.result()
            except concurrent.futures.CancelledError:
                pass
            except Exception as e:
                instance_id = futures[future]
                logger.error(f"Error in future for instance {instance_id}: {e}", exc_info=True)
                progress_manager.on_uncaught_exception(instance_id, e)

    with Live(progress_manager.render_group, refresh_per_second=4):
        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {
                executor.submit(process_instance, instance, output_path, config, progress_manager): instance["instance_id"]
                for instance in instances
            }
            try:
                process_futures(futures)
            except KeyboardInterrupt:
                logger.info("Cancelling all pending jobs. Press ^C again to exit immediately.")
                for future in futures:
                    if not future.running() and not future.done():
                        future.cancel()
                process_futures(futures)


if __name__ == "__main__":
    app()

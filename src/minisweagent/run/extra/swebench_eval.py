#!/usr/bin/env python3

"""Evaluate mini-SWE-agent trajectories for SWE-bench."""
# Read this first: https://mini-swe-agent.com/latest/usage/swebench/  (usage docs)

import concurrent.futures
import json
import threading
import time
import traceback
from pathlib import Path
from typing import Any

import typer
import yaml
from datasets import load_dataset
from rich.live import Live

from minisweagent.config import builtin_config_dir, get_config_path
from minisweagent.models import get_model
from minisweagent.run.extra.swebench import filter_instances, get_sb_environment
from minisweagent.run.extra.utils.batch_progress import RunBatchProgressManager
from minisweagent.utils.log import add_file_handler, logger

_HELP_TEXT = """Evaluate mini-SWE-agent trajectories for SWEBench."""

app = typer.Typer(rich_markup_mode="rich", add_completion=False)


_OUTPUT_FILE_LOCK = threading.Lock()


def update_evals_file(output_path: Path, instance_id: str, model_name: str, model_patch: str, eval_report: dict):
    """Update the output JSON file with results from a single instance."""
    with _OUTPUT_FILE_LOCK:
        output_data = {}
        if output_path.exists():
            output_data = json.loads(output_path.read_text())
        output_data[instance_id] = {
            "model_name_or_path": model_name,
            "instance_id": instance_id,
            "model_patch": model_patch,
            "eval_report": eval_report,
        }
        output_path.write_text(json.dumps(output_data, indent=2))


def remove_from_evals_file(output_path: Path, instance_id: str):
    """Remove an instance from the predictions file."""
    if not output_path.exists():
        return
    with _OUTPUT_FILE_LOCK:
        output_data = json.loads(output_path.read_text())
        if instance_id in output_data:
            del output_data[instance_id]
            output_path.write_text(json.dumps(output_data, indent=2))


def evaluate_instance(
    instance: dict,
    output_dir: Path,
    instance_result: dict[str, Any],
    config: dict,
) -> None:
    """Process a single SWEBench instance."""
    instance_id = instance["instance_id"]
    output_path = output_dir / "evals.json"

    # avoid inconsistent state if something here fails and there's leftover previous files
    remove_from_evals_file(output_path, instance_id)
    model = get_model(config=config.get("model", {}))

    ret = {"instance_id": instance_id, "resolved": False, "eval_error": None}

    env = None
    try:
        env = get_sb_environment(config, instance)
    except Exception as e:
        ret["eval_error"] = f"Env creation failed with {e}"
        logger.info(f"Starting environment failed with exception: {e}\n, {traceback.format_exc()}")
        update_evals_file(output_path, instance_id, model.config.model_name, instance_result["model_patch"], ret)
        return

    # apply git patch
    # NOTE (sumanthrh): This applies patch in-line, and the maximum patch size is limited by the OS limits for `ARG_MAX`.
    # In modern systems, this is typically ~ 1 MB, which is pretty generous.
    # For simplicity, we assume that large patches greater than `ARG_MAX` are meant to fail
    obs = env.execute(f"git apply <<<'EOF'\n{instance_result['model_patch']}\nEOF")

    if obs["returncode"] != 0:
        ret["eval_error"] = obs["output"]
    else:
        # run eval script in-line
        eval_script = instance["eval_script"]
        eval_cmd = f"bash <<'EOF'\n{eval_script}\nEOF"
        obs = env.execute(eval_cmd)
        # use the return value
        ret["resolved"] = obs["returncode"] == 0
        ret["eval_error"] = obs["output"] if not ret["resolved"] else None
    update_evals_file(output_path, instance_id, model.config.model_name, instance_result["model_patch"], ret)


# fmt: off
@app.command(help=_HELP_TEXT)
def main(
    dataset: str = typer.Option("SumanthRH/SWE-Bench_Verified", "--dataset", help="Path to the SWEBench dataset to use. Should include a `eval_script` column specifying the evaluation script for each instance", rich_help_panel="Data selection"),
    split: str = typer.Option("dev", "--split", help="Dataset split", rich_help_panel="Data selection"),
    slice_spec: str = typer.Option("", "--slice", help="Slice specification (e.g., '0:5' for first 5 instances)", rich_help_panel="Data selection"),
    filter_spec: str = typer.Option("", "--filter", help="Filter instance IDs by regex", rich_help_panel="Data selection"),
    shuffle: bool = typer.Option(False, "--shuffle", help="Shuffle instances", rich_help_panel="Data selection"),
    output: str = typer.Option("", "-o", "--output", help="Output directory. Should contain a preds.json file with model predictions", rich_help_panel="Basic"),
    workers: int = typer.Option(1, "-w", "--workers", help="Number of worker threads for parallel processing", rich_help_panel="Basic"),
    model: str | None = typer.Option(None, "-m", "--model", help="Model to use", rich_help_panel="Basic"),
    model_class: str | None = typer.Option(None, "-c", "--model-class", help="Model class to use (e.g., 'anthropic' or 'minisweagent.models.anthropic.AnthropicModel')", rich_help_panel="Advanced"),
    redo_existing: bool = typer.Option(False, "--redo-existing", help="Redo existing instances", rich_help_panel="Data selection"),
    config_spec: Path = typer.Option( builtin_config_dir / "extra" / "swebench_eval.yaml", "-c", "--config", help="Path to a config file", rich_help_panel="Basic"),
    environment_class: str | None = typer.Option( None, "--environment-class", help="Environment type to use. Recommended are docker or singularity", rich_help_panel="Advanced"),
) -> None:
    # fmt: on
    output_path = Path(output)

    predictions_file = output_path / "preds.json"
    if not predictions_file.exists():
        raise FileNotFoundError(f"Expected a `preds.json` file in output directory {output_path}")
    logger.info(f"Results will be saved to {output_path}")
    add_file_handler(output_path / "minisweagent_eval.log")

    logger.info(f"Loading dataset {dataset}, split {split}...")
    instances = list(load_dataset(dataset, split=split))

    instances = filter_instances(instances, filter_spec=filter_spec, slice_spec=slice_spec, shuffle=shuffle)
    if not redo_existing and (output_path / "evals.json").exists():
        existing_instances = list(json.loads((output_path / "evals.json").read_text()).keys())
        logger.info(f"Skipping {len(existing_instances)} existing instances")
        instances = [instance for instance in instances if instance["instance_id"] not in existing_instances]
    logger.info(f"Running on {len(instances)} instances...")

    with open(output_path / "preds.json") as f:
        predictions = json.load(f)

    config = yaml.safe_load(get_config_path(config_spec).read_text())
    if environment_class is not None:
        config.setdefault("environment", {})["environment_class"] = environment_class
    if model is not None:
        config.setdefault("model", {})["model_name"] = model
    if model_class is not None:
        config.setdefault("model", {})["model_class"] = model_class

    progress_manager = RunBatchProgressManager(len(instances), output_path / f"exit_statuses_eval_{time.time()}.yaml")

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
                executor.submit(evaluate_instance, instance, output_path, predictions[instance["instance_id"]], config): instance[
                    "instance_id"
                ]
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

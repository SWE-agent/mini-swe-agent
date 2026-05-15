#!/usr/bin/env python3
"""Run PlanMemAgent on SWE-bench instances in batch mode.

Mirrors ``minisweagent.run.benchmarks.swebench`` (the official runner) but
uses :class:`PlanMemAgent` instead of ``DefaultAgent``. Reuses every helper
from the official module so we stay in lockstep with v2 conventions.

Usage:
    python -m minisweagent.run.benchmarks.swebench_planmem \\
        --subset verified --split test --workers 4 \\
        -c src/minisweagent/config/benchmarks/swebench_planmem.yaml \\
        --model openai/Qwen/Qwen3-Coder-30B-A3B-Instruct-FP8 \\
        --model-class litellm \\
        -o experiments/outputs/planmem_smoke

For the recommended ``mini-extra``-style invocation we expose a Typer app
identical in shape to the official one, just bound to a PlanMem
``ProgressTrackingAgent`` subclass. We deliberately do NOT monkey-patch the
upstream runner.
"""

from __future__ import annotations

import concurrent.futures
import json
import time
import traceback
from pathlib import Path

import typer
from rich.live import Live

from minisweagent.agents.planmem_agent import PlanMemAgent
from minisweagent.config import builtin_config_dir, get_config_from_spec
from minisweagent.models import get_model
from minisweagent.run.benchmarks.swebench import (
    _OUTPUT_FILE_LOCK,  # noqa: F401  (kept for parity if the official module changes signature)
    DATASET_MAPPING,
    filter_instances,
    get_sb_environment,
    remove_from_preds_file,
    update_preds_file,
)
from minisweagent.run.benchmarks.utils.batch_progress import RunBatchProgressManager
from minisweagent.utils.log import add_file_handler, logger
from minisweagent.utils.serialize import UNSET, recursive_merge

DEFAULT_CONFIG_FILE = builtin_config_dir / "benchmarks" / "swebench_planmem.yaml"


class ProgressTrackingPlanMemAgent(PlanMemAgent):
    """PlanMemAgent + per-step progress callback (mirror of the official wrapper)."""

    def __init__(
        self,
        *args,
        progress_manager: RunBatchProgressManager,
        instance_id: str = "",
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.progress_manager = progress_manager
        self.instance_id = instance_id

    def step(self) -> dict:
        phase = "?"
        if self._planning_signal is not None:
            phase = self._planning_signal.current_phase.value[:5]
        self.progress_manager.update_instance_status(
            self.instance_id,
            f"Step {self.n_calls + 1:3d} ({phase}, ${self.cost:.2f})",
        )
        return super().step()


def process_instance(
    instance: dict,
    output_dir: Path,
    config: dict,
    progress_manager: RunBatchProgressManager,
) -> None:
    """Process a single SWEBench instance with PlanMemAgent."""
    instance_id = instance["instance_id"]
    instance_dir = output_dir / instance_id
    remove_from_preds_file(output_dir / "preds.json", instance_id)
    (instance_dir / f"{instance_id}.traj.json").unlink(missing_ok=True)

    model = get_model(config=config.get("model", {}))
    task = instance["problem_statement"]

    progress_manager.on_instance_start(instance_id)
    progress_manager.update_instance_status(instance_id, "Pulling/starting environment")

    agent = None
    exit_status, result = None, None
    extra_info: dict = {}

    try:
        env = get_sb_environment(config, instance)
        agent_kwargs = dict(config.get("agent", {}))
        # ``agent_class`` is read at the alias layer; strip it so it doesn't
        # bleed into the Pydantic config and trigger an unknown-field error.
        agent_kwargs.pop("agent_class", None)
        agent = ProgressTrackingPlanMemAgent(
            model,
            env,
            progress_manager=progress_manager,
            instance_id=instance_id,
            **agent_kwargs,
        )
        info = agent.run(task)
        exit_status = info.get("exit_status")
        result = info.get("submission")
    except Exception as e:
        logger.error(f"Error processing instance {instance_id}: {e}", exc_info=True)
        exit_status, result = type(e).__name__, ""
        extra_info = {"traceback": traceback.format_exc(), "exception_str": str(e)}
    finally:
        if agent is not None:
            traj_path = instance_dir / f"{instance_id}.traj.json"
            agent.save(
                traj_path,
                {
                    "info": {
                        "exit_status": exit_status,
                        "submission": result,
                        **extra_info,
                    },
                    "instance_id": instance_id,
                    "planner_progress": agent.planner.progress if agent.config.enable_planner else "disabled",
                },
            )
            logger.info(f"Saved trajectory to '{traj_path}'")
        update_preds_file(
            output_dir / "preds.json",
            instance_id,
            model.config.model_name,
            result,
        )
        progress_manager.on_instance_end(instance_id, exit_status)


app = typer.Typer(rich_markup_mode="rich", add_completion=False)


# fmt: off
@app.command(help="Run PlanMemAgent on SWE-bench (mirrors mini-extra swebench).")
def main(
    subset: str = typer.Option("verified", "--subset", help="SWE-bench subset"),
    split: str = typer.Option("test", "--split", help="Dataset split"),
    slice_spec: str = typer.Option("", "--slice", help="Slice spec, e.g. '0:5'"),
    filter_spec: str = typer.Option("", "--filter", help="Regex filter on instance_id"),
    shuffle: bool = typer.Option(False, "--shuffle", help="Shuffle instances"),
    output: str = typer.Option("", "-o", "--output", help="Output directory"),
    workers: int = typer.Option(1, "-w", "--workers", help="Worker threads"),
    model: str | None = typer.Option(None, "-m", "--model", help="Model name"),
    model_class: str | None = typer.Option(None, "--model-class", help="Model class alias"),
    redo_existing: bool = typer.Option(False, "--redo-existing"),
    config_spec: list[str] = typer.Option([str(DEFAULT_CONFIG_FILE)], "-c", "--config"),
    environment_class: str | None = typer.Option(None, "--environment-class"),
) -> None:
    # fmt: on
    if not output:
        raise typer.BadParameter("--output / -o is required")
    output_path = Path(output)
    output_path.mkdir(parents=True, exist_ok=True)
    logger.info(f"Results will be saved to {output_path}")
    add_file_handler(output_path / "minisweagent.log")

    from datasets import load_dataset

    dataset_path = DATASET_MAPPING.get(subset, subset)
    logger.info(f"Loading dataset {dataset_path}, split {split}...")
    instances = list(load_dataset(dataset_path, split=split))

    instances = filter_instances(
        instances, filter_spec=filter_spec, slice_spec=slice_spec, shuffle=shuffle,
    )
    if not redo_existing and (output_path / "preds.json").exists():
        existing = list(json.loads((output_path / "preds.json").read_text()).keys())
        logger.info(f"Skipping {len(existing)} existing instances")
        instances = [i for i in instances if i["instance_id"] not in existing]
    logger.info(f"Running on {len(instances)} instances...")

    logger.info(f"Building agent config from specs: {config_spec}")
    configs = [get_config_from_spec(spec) for spec in config_spec]
    configs.append({
        "environment": {"environment_class": environment_class or UNSET},
        "model": {"model_name": model or UNSET, "model_class": model_class or UNSET},
    })
    config = recursive_merge(*configs)

    progress_manager = RunBatchProgressManager(
        len(instances), output_path / f"exit_statuses_{time.time()}.yaml",
    )

    def process_futures(futures: dict[concurrent.futures.Future, str]) -> None:
        for future in concurrent.futures.as_completed(futures):
            try:
                future.result()
            except concurrent.futures.CancelledError:
                pass
            except Exception as e:
                instance_id = futures[future]
                logger.error(
                    f"Error in future for instance {instance_id}: {e}", exc_info=True,
                )
                progress_manager.on_uncaught_exception(instance_id, e)

    with Live(progress_manager.render_group, refresh_per_second=4):
        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {
                executor.submit(
                    process_instance, instance, output_path, config, progress_manager,
                ): instance["instance_id"]
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

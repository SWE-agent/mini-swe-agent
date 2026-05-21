"""Run mini-SWE-agent on ProgramBench instances in batch mode."""

import concurrent.futures
import copy
import subprocess
import time
import traceback
from pathlib import Path

import typer
from rich.live import Live

from minisweagent.config import builtin_config_dir, get_config_from_spec
from minisweagent.environments import get_environment
from minisweagent.models import get_model
from minisweagent.run.benchmarks.utils.batch_progress import RunBatchProgressManager
from minisweagent.run.benchmarks.utils.common import ProgressTrackingAgent
from minisweagent.utils.log import add_file_handler, logger
from minisweagent.utils.serialize import UNSET, recursive_merge

_HELP_TEXT = """Run mini-SWE-agent on ProgramBench instances.

Requires [bold green]programbench[/bold green] to be installed ([bold]pip install programbench[/bold]).
Output is compatible with [bold green]programbench eval[/bold green].
"""

DEFAULT_CONFIG_FILE = builtin_config_dir / "benchmarks" / "programbench.yaml"
_IMAGE_TAG = "task_cleanroom"

app = typer.Typer(rich_markup_mode="rich", add_completion=False)


class ProgramBenchAgent(ProgressTrackingAgent):
    """Drops ``raw_output`` from tool-result messages to avoid bloating trajectories."""

    def serialize(self, *extra_dicts) -> dict:
        data = super().serialize(*extra_dicts)
        for msg in data.get("messages", []):
            extra = msg.get("extra", {})
            extra.pop("raw_output", None)
            for obs in extra.get("observations", []):
                obs.pop("raw_output", None)
        return data


def copy_submission(env, dest: Path, *, src: str = "/workspace") -> None:
    """Tar+gzip the workspace from the container to a local file."""
    container_id = getattr(env, "container_id", None)
    executable = getattr(getattr(env, "config", None), "executable", None)
    if not container_id or not executable:
        raise RuntimeError("copy_submission requires a Docker environment with container_id")
    dest.parent.mkdir(parents=True, exist_ok=True)
    container_tar = "/tmp/_submission.tar.gz"
    env.execute({"command": f"tar -czf {container_tar} -C {src} ."})
    subprocess.run(
        [executable, "cp", f"{container_id}:{container_tar}", str(dest)],
        check=True,
        capture_output=True,
        text=True,
    )


def process_instance(
    instance: dict,
    output_dir: Path,
    config: dict,
    progress_manager: RunBatchProgressManager,
) -> None:
    """Process a single ProgramBench instance."""
    iid = instance["instance_id"]
    instance_dir = output_dir / iid
    (instance_dir / f"{iid}.traj.json").unlink(missing_ok=True)

    progress_manager.on_instance_start(iid)
    progress_manager.update_instance_status(iid, "Starting environment")

    inst_config = copy.deepcopy(config)
    inst_config.setdefault("environment", {})["image"] = f"{instance['image_name']}:{_IMAGE_TAG}"

    agent = None
    exit_status = None
    extra_info: dict = {}

    try:
        model = get_model(config=inst_config.get("model", {}))
        env = get_environment(inst_config.get("environment", {}), default_type="docker")
        env.execute(
            {"command": 'git config user.name "mini-swe-agent" && git config user.email "mini-swe-agent@proton.me"'}
        )

        agent_config = dict(inst_config.get("agent", {}))
        agent_config["output_path"] = str(instance_dir / f"{iid}.traj.json")
        agent = ProgramBenchAgent(
            model,
            env,
            progress_manager=progress_manager,
            instance_id=iid,
            **agent_config,
        )
        agent.extra_template_vars = {"instance": instance}
        info = agent.run()
        exit_status = info.get("exit_status")
    except Exception as e:
        logger.error(f"Error processing instance {iid}: {e}", exc_info=True)
        exit_status = type(e).__name__
        extra_info = {"traceback": traceback.format_exc(), "exception_str": str(e)}
    finally:
        if agent is not None:
            try:
                copy_submission(agent.env, instance_dir / "submission.tar.gz")
            except Exception as e:
                logger.error(f"Failed to copy submission for {iid}: {e}", exc_info=True)
                extra_info["submission_copy_error"] = str(e)
            traj_path = instance_dir / f"{iid}.traj.json"
            agent.save(traj_path, {"info": {"exit_status": exit_status, **extra_info}, "instance_id": iid})
            logger.info(f"Saved trajectory to '{traj_path}'")
        progress_manager.on_instance_end(iid, exit_status)


# fmt: off
@app.command(help=_HELP_TEXT)
def main(
    slice_spec: str = typer.Option("", "--slice", help="Slice specification (e.g., '0:5' for first 5 instances)", rich_help_panel="Data selection"),
    filter_spec: str = typer.Option("", "--filter", help="Filter instance IDs by regex", rich_help_panel="Data selection"),
    shuffle: bool = typer.Option(False, "--shuffle", help="Shuffle instances", rich_help_panel="Data selection"),
    output: str = typer.Option("", "-o", "--output", help="Output directory", rich_help_panel="Basic"),
    workers: int = typer.Option(1, "-w", "--workers", help="Number of worker threads for parallel processing", rich_help_panel="Basic"),
    model: str | None = typer.Option(None, "-m", "--model", help="Model to use", rich_help_panel="Basic"),
    model_class: str | None = typer.Option(None, "--model-class", help="Model class to use", rich_help_panel="Advanced"),
    redo_existing: bool = typer.Option(False, "--redo-existing", help="Redo existing instances", rich_help_panel="Data selection"),
    config_spec: list[str] = typer.Option([str(DEFAULT_CONFIG_FILE)], "-c", "--config", help="Config files (merged left to right)", rich_help_panel="Basic"),
    environment_class: str | None = typer.Option(None, "--environment-class", help="Environment type (e.g., docker, singularity)", rich_help_panel="Advanced"),
) -> None:
    # fmt: on
    from programbench.utils.instance_filters import filter_instances  # pylint: disable=import-error
    from programbench.utils.load_data import load_all_instances  # pylint: disable=import-error

    output_path = Path(output) if output else Path(f"programbench_results_{int(time.time())}")
    output_path.mkdir(parents=True, exist_ok=True)
    logger.info(f"Results will be saved to {output_path}")
    add_file_handler(output_path / "minisweagent.log")

    instances = load_all_instances(include_tests=False)
    instances = filter_instances(instances, filter_spec=filter_spec, slice_spec=slice_spec, shuffle=shuffle)

    if not redo_existing:
        existing = {i["instance_id"] for i in instances if (output_path / i["instance_id"] / "submission.tar.gz").exists()}
        if existing:
            logger.info(f"Skipping {len(existing)} existing instances")
            instances = [i for i in instances if i["instance_id"] not in existing]

    logger.info(f"Running on {len(instances)} instances...")

    configs = [get_config_from_spec(spec) for spec in config_spec]
    configs.append({
        "environment": {"environment_class": environment_class or UNSET},
        "model": {"model_name": model or UNSET, "model_class": model_class or UNSET},
    })
    config = recursive_merge(*configs)

    progress_manager = RunBatchProgressManager(len(instances), output_path / f"exit_statuses_{int(time.time())}.yaml")

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
                executor.submit(process_instance, instance, output_path, config, progress_manager): instance[
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

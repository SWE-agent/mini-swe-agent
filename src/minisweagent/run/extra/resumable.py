"""Start or resume a controlled local run."""

from pathlib import Path

import typer

from minisweagent.agents.extra.resumable import ResumableAgent
from minisweagent.agents.utils.checkpoint import PAUSE_STATUSES, load_checkpoint
from minisweagent.config import get_config_from_spec
from minisweagent.environments.local import LocalEnvironment
from minisweagent.models import get_model
from minisweagent.utils.serialize import recursive_merge

app = typer.Typer(help=__doc__)
MODEL_TYPES = {
    "litellm": "minisweagent.models.litellm_model.LitellmModel",
    "litellm_textbased": "minisweagent.models.litellm_textbased_model.LitellmTextbasedModel",
}


def make_agent(config: dict, run_dir: Path, pause_after: int, **budgets) -> ResumableAgent:
    model_config = config.get("model", {})
    model_class = model_config.get("model_class", "litellm")
    if model_class not in MODEL_TYPES and model_class not in MODEL_TYPES.values():
        raise ValueError("Controlled resume supports litellm and litellm_textbased only")
    if config.get("environment", {}).get("environment_class", "local") != "local":
        raise ValueError("Controlled resume requires the local environment")
    if {"agent_class", "mode"} & config["agent"].keys():
        typer.echo(
            "Controlled resume ignores agent_class/mode; commands run without interactive confirmation.", err=True
        )
    return ResumableAgent(
        get_model(config=model_config),
        LocalEnvironment(**config.get("environment", {})),
        run_dir=run_dir,
        pause_after=pause_after,
        **(config["agent"] | {key: value for key, value in budgets.items() if value is not None}),
    )


def report_result(result: dict, run_dir: Path) -> None:
    typer.echo(f"Exit status: {result['exit_status']}\nSaved run to {run_dir.resolve()}")
    if result["exit_status"] not in PAUSE_STATUSES | {"Submitted"}:
        raise typer.Exit(code=1)


@app.command()
def start(
    task: str = typer.Option(..., "--task", "-t"),
    run_dir: Path = typer.Option(..., "--run-dir"),
    cwd: Path = typer.Option(Path("."), "--cwd", exists=True, file_okay=False),
    config: list[str] = typer.Option(["mini.yaml"], "--config", "-c"),
    pause_after: int = typer.Option(0, min=0, help="Pause after this many calls in this session (0 disables)."),
    step_limit: int | None = typer.Option(None, min=0, help="Cumulative call limit (0 disables)."),
    cost_limit: float | None = typer.Option(None, min=0, help="Cumulative cost limit (0 disables)."),
    wall_time_limit_seconds: int | None = typer.Option(None, min=0, help="Cumulative active time limit (0 disables)."),
) -> None:
    """Start a new task. Commands execute locally without interactive confirmation."""
    settings = recursive_merge(*[get_config_from_spec(spec) for spec in config])
    settings.setdefault("environment", {})["cwd"] = str(cwd.resolve())
    report_result(
        make_agent(
            settings,
            run_dir,
            pause_after,
            step_limit=step_limit,
            cost_limit=cost_limit,
            wall_time_limit_seconds=wall_time_limit_seconds,
        ).run(task),
        run_dir,
    )


@app.command()
def resume(
    run_dir: Path = typer.Option(..., "--run-dir", exists=True, file_okay=False),
    pause_after: int = typer.Option(0, min=0, help="Pause after this many additional calls (0 disables)."),
    step_limit: int | None = typer.Option(None, min=0, help="New cumulative call limit (0 disables)."),
    cost_limit: float | None = typer.Option(None, min=0, help="New cumulative cost limit (0 disables)."),
    wall_time_limit_seconds: int | None = typer.Option(
        None, min=0, help="New cumulative active time limit (0 disables)."
    ),
) -> None:
    """Continue the latest controlled pause, using its saved task and configuration."""
    data, _ = load_checkpoint(run_dir)
    settings = data["info"]["config"]
    settings["model"]["model_class"] = settings["model_type"]
    settings["agent"].pop("output_path")
    report_result(
        make_agent(
            settings,
            run_dir,
            pause_after,
            step_limit=step_limit,
            cost_limit=cost_limit,
            wall_time_limit_seconds=wall_time_limit_seconds,
        ).resume(),
        run_dir,
    )


if __name__ == "__main__":
    app()

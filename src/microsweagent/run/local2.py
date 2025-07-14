#!/usr/bin/env python3
import os
from pathlib import Path

import typer
import yaml

from microsweagent import package_dir
from microsweagent.agents.interactive_textual import AgentApp
from microsweagent.environments.local import LocalEnvironment
from microsweagent.models import get_model
from microsweagent.run.local import get_multiline_task
from microsweagent.run.utils.save import save_traj

DEFAULT_CONFIG = Path(os.getenv("MSWEA_LOCAL2_CONFIG_PATH", package_dir / "config" / "local.yaml"))
app = typer.Typer()


@app.command()
def main(
    config: str = typer.Option(str(DEFAULT_CONFIG), "-c", "--config", help="Path to config file"),
    model: str | None = typer.Option(
        None,
        "-m",
        "--model",
        help="Model to use",
    ),
    task: str | None = typer.Option(None, "-t", "--task", help="Task/problem statement", show_default=False),
    yolo: bool = typer.Option(False, "-y", "--yolo", help="Run without confirmation"),
) -> None:
    """Run micro-SWE-agent with textual interface."""
    _config = yaml.safe_load(Path(config).read_text())

    if not task:
        task = get_multiline_task()
        if not task:
            typer.echo("No task/problem statement provided.")
            raise typer.Exit(1)

    if yolo:
        _config["agent"]["mode"] = "yolo"

    # Create and run the agent app
    agent_app = AgentApp(
        model=get_model(model, _config.get("model", {})),
        env=LocalEnvironment(),
        task=task,
        **_config.get("agent", {}),
    )

    try:
        agent_app.run()
    except KeyboardInterrupt:
        typer.echo("\nKeyboardInterrupt -- goodbye")
    finally:
        save_traj(agent_app.agent, Path("traj.json"), exit_status=agent_app.exit_status, result=agent_app.result)


def cli() -> None:
    """CLI entry point for script execution."""
    app()


if __name__ == "__main__":
    app()

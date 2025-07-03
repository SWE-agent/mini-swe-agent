#!/usr/bin/env python3
import json
import os
from pathlib import Path

import typer
import yaml
from rich.console import Console

from microswea import package_dir
from microswea.agents.micro import Agent
from microswea.environments.local import LocalEnvironment
from microswea.models.litellm_model import LitellmModel

DEFAULT_CONFIG = Path(os.getenv("MSWEA_LOCAL_CONFIG_PATH", package_dir / "config" / "local.yaml"))
console = Console(highlight=False)
app = typer.Typer()


def get_multiline_problem_statement() -> str:
    """Get a multiline problem statement from the user."""
    console.print(
        "[bold yellow]What do you want to do?[/bold yellow] (press [bold red]Ctrl+D[/bold red] in a [red]new line[/red] to finish)"
    )
    lines = []
    while True:
        try:
            lines.append(console.input("[bold green]>[/bold green] "))
        except EOFError:
            break

    return "\n".join(lines).strip()


@app.command()
def main(
    config: str = typer.Option(str(DEFAULT_CONFIG), "--config", help="Path to config file"),
    model: str | None = typer.Option(
        None,
        "--model",
        help="Model to use",
    ),
    problem: str | None = typer.Option(None, "--problem", help="Problem statement", show_default=False),
    yolo: bool = typer.Option(False, "--yolo", help="Run without confirmation"),
) -> Agent:
    """Run micro-SWE-agent right here, right now."""
    _config = yaml.safe_load(Path(config).read_text())
    _model = _config.get("model", {}).get("model_name")
    if model:
        _model = model
    if not _model:
        _model = os.getenv("MSWEA_MODEL_NAME")
    if not _model:
        _model = console.input("[bold yellow]Enter your model name: [/bold yellow]")

    if not problem:
        problem = get_multiline_problem_statement()

    agent = Agent(
        LitellmModel(**(_config.get("model", {}) | {"model_name": _model})),
        LocalEnvironment(),
        problem,
        **(_config["agent"] | {"confirm_actions": not yolo}),
    )

    try:
        result = agent.run()
    except KeyboardInterrupt:
        console.print("\n[bold red]KeyboardInterrupt -- goodbye[/bold red]")
    else:
        Path("patch.txt").write_text(result)
    finally:
        Path("traj.json").write_text(
            json.dumps(agent.history, indent=2),
        )
        console.print(f"Total cost: [bold green]${agent.model.cost:.4f}[/bold green]")
        console.print(f"Total steps: [bold green]{agent.model.n_calls}[/bold green]")
    return agent


if __name__ == "__main__":
    app()

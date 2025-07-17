#!/usr/bin/env python3

import os
from pathlib import Path
from typing import Any

import typer
import yaml
from rich.console import Console

from microsweagent import Environment, Model
from microsweagent.agents.interactive import InteractiveAgent
from microsweagent.agents.interactive_textual import AgentApp
from microsweagent.config import builtin_config_dir, get_config_path
from microsweagent.environments.local import LocalEnvironment
from microsweagent.models import get_model
from microsweagent.run.utils.save import save_traj

DEFAULT_CONFIG = Path(os.getenv("MSWEA_LOCAL_CONFIG_PATH", builtin_config_dir / "local.yaml"))
console = Console(highlight=False)
app = typer.Typer()


def get_multiline_task() -> str:
    """Get a multiline task/problem statement from the user."""
    console.print(
        "[bold yellow]What do you want to do?[/bold yellow] (press [bold red]Ctrl+D[/bold red] in a [red]new line[/red] to finish)"
    )
    lines = []
    while True:
        try:
            lines.append(console.input("[bold green]>[/bold green] "))
        except EOFError:
            break

    print()
    return "\n".join(lines).strip()


def run_interactive(model: Model, env: Environment, agent_config: dict, task: str, output: Path | None = None) -> Any:
    agent = InteractiveAgent(
        model,
        env,
        **agent_config,
    )

    exit_status, result = None, None
    try:
        exit_status, result = agent.run(task)
    except KeyboardInterrupt:
        console.print("\n[bold red]KeyboardInterrupt -- goodbye[/bold red]")
    finally:
        if output:
            save_traj(agent, output, exit_status=exit_status, result=result)
    return agent


def run_textual(model: Model, env: Environment, agent_config: dict, task: str, output: Path | None = None) -> Any:
    agent_app = AgentApp(
        model,
        env,
        task,
        **agent_config,
    )
    try:
        agent_app.run()
    except KeyboardInterrupt:
        typer.echo("\nKeyboardInterrupt -- goodbye")
    finally:
        save_traj(agent_app.agent, Path("traj.json"), exit_status=agent_app.exit_status, result=agent_app.result)


@app.command()
def main(
    config: Path = typer.Option(DEFAULT_CONFIG, "-c", "--config", help="Path to config file"),
    model: str | None = typer.Option(
        None,
        "-m",
        "--model",
        help="Model to use",
    ),
    task: str | None = typer.Option(None, "-t", "--task", help="Task/problem statement", show_default=False),
    yolo: bool = typer.Option(False, "-y", "--yolo", help="Run without confirmation"),
    output: Path | None = typer.Option(None, "-o", "--output", help="Output file"),
    visual: bool = typer.Option(False, "-v", "--visual", help="Use visual UI (Textual)"),
) -> Any:
    """Run micro-SWE-agent right here, right now."""
    _config = yaml.safe_load(get_config_path(config).read_text())

    if not task:
        task = get_multiline_task()

    mode = "confirm" if not yolo else "yolo"

    _model = get_model(model, _config.get("model", {}))
    env = LocalEnvironment(**_config.get("env", {}))
    agent_config = _config.get("agent", {}) | {"mode": mode}

    if visual:
        return run_textual(_model, env, agent_config, task, output)
    else:
        return run_interactive(_model, env, agent_config, task, output)


if __name__ == "__main__":
    app()

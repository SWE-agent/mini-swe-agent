#!/usr/bin/env python3

import os
from pathlib import Path
from typing import Any

import typer
import yaml
from prompt_toolkit.history import FileHistory
from prompt_toolkit.shortcuts import PromptSession
from rich.console import Console

from microsweagent import Environment, Model, global_config_dir
from microsweagent.agents.interactive import InteractiveAgent
from microsweagent.agents.interactive_textual import AgentApp
from microsweagent.config import builtin_config_dir, get_config_path
from microsweagent.environments.local import LocalEnvironment
from microsweagent.models import get_model
from microsweagent.run.utils.save import save_traj

DEFAULT_CONFIG = Path(os.getenv("MSWEA_LOCAL_CONFIG_PATH", builtin_config_dir / "local.yaml"))
console = Console(highlight=False)
app = typer.Typer()
prompt_session = PromptSession(history=FileHistory(global_config_dir / "micro_task_history.txt"))


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
    config_spec: Path = typer.Option(DEFAULT_CONFIG, "-c", "--config", help="Path to config file"),
    model_name: str | None = typer.Option(
        None,
        "-m",
        "--model",
        help="Model to use",
    ),
    task: str | None = typer.Option(None, "-t", "--task", help="Task/problem statement", show_default=False),
    yolo: bool = typer.Option(False, "-y", "--yolo", help="Run without confirmation"),
    output: Path | None = typer.Option(None, "-o", "--output", help="Output file"),
    visual: bool = typer.Option(False, "-v", "--visual", help="Use visual UI (Textual)"),
    cost_limit: float | None = typer.Option(None, "-l", "--cost-limit", help="Cost limit. Set to 0 to disable."),
) -> Any:
    """Run micro-SWE-agent right here, right now."""
    config = yaml.safe_load(get_config_path(config_spec).read_text())

    if not task:
        console.print(
            "[bold yellow]What do you want to do?\n"
            "[bold green]Up[/bold green]/[bold green]Down[/bold green] to bring up previous tasks or [bold green]Ctrl+R[/bold green] to search\n"
            "Confirm input with [bold green]Esc, Enter[/bold green]\n"
            ">[/bold yellow] ",
            end="",
        )
        task = prompt_session.prompt("", multiline=True)
        console.print("[bold green]Got that, thanks![/bold green]")

    config["agent"]["mode"] = "confirm" if not yolo else "yolo"
    if cost_limit:
        config["agent"]["cost_limit"] = cost_limit
    model = get_model(model_name, config.get("model", {}))
    env = LocalEnvironment(**config.get("env", {}))

    if visual:
        return run_textual(model, env, config["agent"], task, output)  # type: ignore[arg-type]
    else:
        return run_interactive(model, env, config["agent"], task, output)  # type: ignore[arg-type]


if __name__ == "__main__":
    app()

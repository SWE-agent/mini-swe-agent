#!/usr/bin/env python3

"""Run micro-SWE-agent in your local environment.

[not dim]
There are two different user interfaces:

[bold green]micro[/bold green] Simple REPL-style interface
[bold green]micro -v[/bold green] Pager-style interface (Textual)

More information about the usage: [bold green]https://micro-swe-agent.com/latest/usage/micro/[/bold green]
[/not dim]
"""

import os
from pathlib import Path
from typing import Any

import typer
import yaml
from prompt_toolkit.formatted_text import HTML
from prompt_toolkit.history import FileHistory
from prompt_toolkit.shortcuts import PromptSession
from rich.console import Console

from microsweagent import Environment, Model, global_config_dir
from microsweagent.agents.interactive import InteractiveAgent
from microsweagent.agents.interactive_textual import AgentApp
from microsweagent.config import builtin_config_dir, get_config_path
from microsweagent.environments.local import LocalEnvironment
from microsweagent.models import get_model
from microsweagent.run.extra.config import configure_if_first_time
from microsweagent.run.utils.save import save_traj

DEFAULT_CONFIG = Path(os.getenv("MSWEA_LOCAL_CONFIG_PATH", builtin_config_dir / "local.yaml"))
console = Console(highlight=False)
app = typer.Typer(rich_markup_mode="rich")
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


@app.command(help=__doc__)
def main(
    visual: bool = typer.Option(False, "-v", "--visual", help="Use visual (pager-style) UI (Textual)"),
    model_name: str | None = typer.Option(
        None,
        "-m",
        "--model",
        help="Model to use",
    ),
    task: str | None = typer.Option(None, "-t", "--task", help="Task/problem statement", show_default=False),
    yolo: bool = typer.Option(False, "-y", "--yolo", help="Run without confirmation"),
    cost_limit: float | None = typer.Option(None, "-l", "--cost-limit", help="Cost limit. Set to 0 to disable."),
    config_spec: Path = typer.Option(DEFAULT_CONFIG, "-c", "--config", help="Path to config file"),
    output: Path | None = typer.Option(None, "-o", "--output", help="Output file"),
) -> Any:
    configure_if_first_time()
    config = yaml.safe_load(get_config_path(config_spec).read_text())

    if not task:
        console.print("[bold yellow]What do you want to do?")
        task = prompt_session.prompt(
            "",
            multiline=True,
            bottom_toolbar=HTML(
                "Submit task: <b fg='yellow' bg='black'>Esc+Enter</b> | "
                "Navigate history: <b fg='yellow' bg='black'>Arrow Up/Down</b> | "
                "Search history: <b fg='yellow' bg='black'>Ctrl+R</b>"
            ),
        )
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

#!/usr/bin/env python3
import os
from pathlib import Path

import requests
import typer
from rich.console import Console

from minisweagent.agents.interactive import InteractiveAgent
from minisweagent.config import builtin_config_dir, get_config_from_spec
from minisweagent.environments.docker import DockerEnvironment
from minisweagent.models import get_model
from minisweagent.run.extra.config import configure_if_first_time
from minisweagent.utils.serialize import recursive_merge

DEFAULT_CONFIG = Path(os.getenv("MSWEA_GITHUB_CONFIG_PATH", builtin_config_dir / "github_issue.yaml"))
console = Console(highlight=False)
app = typer.Typer(rich_markup_mode="rich", add_completion=False)

_CONFIG_SPEC_HELP_TEXT = """Path to config files, filenames, or key-value pairs.

[bold red]IMPORTANT:[/bold red] [red]If you set this option, the default config file will not be used.[/red]
So you need to explicitly set it e.g., with [bold green]-c github_issue.yaml <other options>[/bold green]

Multiple configs will be recursively merged.

Examples:

[bold red]-c model.model_kwargs.temperature=0[/bold red] [red]You forgot to add the default config file! See above.[/red]

[bold green]-c github_issue.yaml -c model.model_kwargs.temperature=0.5[/bold green]

[bold green]-c github_issue.yaml -c agent.mode=yolo[/bold green]
"""


def fetch_github_issue(issue_url: str) -> str:
    """Fetch GitHub issue text from the URL."""
    # Convert GitHub issue URL to API URL
    api_url = issue_url.replace("github.com", "api.github.com/repos").replace("/issues/", "/issues/")

    headers = {}
    if github_token := os.getenv("GITHUB_TOKEN"):
        headers["Authorization"] = f"token {github_token}"

    response = requests.get(api_url, headers=headers)
    issue_data = response.json()

    title = issue_data["title"]
    body = issue_data["body"] or ""

    return f"GitHub Issue: {title}\n\n{body}"


# fmt: off
@app.command()
def main(
    issue_url: str = typer.Option(prompt="Enter GitHub issue URL", help="GitHub issue URL"),
    config_spec: list[str] = typer.Option([str(DEFAULT_CONFIG)], "-c", "--config", help=_CONFIG_SPEC_HELP_TEXT),
    model: str | None = typer.Option(None, "-m", "--model", help="Model to use"),
    model_class: str | None = typer.Option(None, "--model-class", help="Model class to use (e.g., 'anthropic' or 'minisweagent.models.anthropic.AnthropicModel')", rich_help_panel="Advanced"),
    yolo: bool = typer.Option(False, "-y", "--yolo", help="Run without confirmation"),
) -> InteractiveAgent:
    # fmt: on
    """Run mini-SWE-agent on a GitHub issue"""
    configure_if_first_time()

    console.print(f"Building agent config from specs: [bold green]{config_spec}[/bold green]")
    configs = [get_config_from_spec(spec) for spec in config_spec]
    if yolo:
        configs.append({"agent": {"mode": "yolo"}})
    if model_class is not None:
        configs.append({"model": {"model_class": model_class}})
    if model is not None:
        configs.append({"model": {"model_name": model}})
    config = recursive_merge(*configs)

    task = fetch_github_issue(issue_url)

    env = DockerEnvironment(**config.get("environment", {}))

    repo_url = issue_url.split("/issues/")[0]
    if github_token := os.getenv("GITHUB_TOKEN"):
        repo_url = repo_url.replace("https://github.com/", f"https://{github_token}@github.com/") + ".git"

    env.execute({"command": f"git clone {repo_url} /testbed"}, cwd="/")

    agent = InteractiveAgent(
        get_model(config=config.get("model", {})),
        env,
        **config.get("agent", {}),
    )

    agent.run(task)
    return agent


if __name__ == "__main__":
    app()

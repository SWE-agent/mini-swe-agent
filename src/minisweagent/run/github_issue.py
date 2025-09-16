#!/usr/bin/env python3
import os
from pathlib import Path

import requests
import typer
import yaml
from rich.console import Console

from minisweagent.agents.interactive import InteractiveAgent
from minisweagent.config import builtin_config_dir, get_config_path
from minisweagent.environments.docker import DockerEnvironment
from minisweagent.models import get_model
from minisweagent.run.extra.config import configure_if_first_time
from minisweagent.run.utils.save import save_traj

DEFAULT_CONFIG = Path(os.getenv("MSWEA_GITHUB_CONFIG_PATH", builtin_config_dir / "github_issue.yaml"))
console = Console(highlight=False)
app = typer.Typer(rich_markup_mode="rich", add_completion=False)


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
    config: Path = typer.Option(DEFAULT_CONFIG, "-c", "--config", help="Path to config file"),
    model: str | None = typer.Option(None, "-m", "--model", help="Model to use"),
    model_class: str | None = typer.Option(None, "--model-class", help="Model class to use (e.g., 'anthropic' or 'minisweagent.models.anthropic.AnthropicModel')", rich_help_panel="Advanced"),
    yolo: bool = typer.Option(False, "-y", "--yolo", help="Run without confirmation"),
) -> InteractiveAgent:
    # fmt: on
    """Run mini-SWE-agent on a GitHub issue"""
    configure_if_first_time()

    config_path = get_config_path(config)
    console.print(f"Loading agent config from [bold green]'{config_path}'[/bold green]")
    _config = yaml.safe_load(config_path.read_text())
    _agent_config = _config.setdefault("agent", {})
    if yolo:
        _agent_config["mode"] = "yolo"
    if model_class is not None:
        _config.setdefault("model", {})["model_class"] = model_class

    task = fetch_github_issue(issue_url)

    agent = InteractiveAgent(
        get_model(model, _config.get("model", {})),
        DockerEnvironment(**_config.get("environment", {})),
        **_agent_config,
    )

    repo_url = issue_url.split("/issues/")[0]
    if github_token := os.getenv("GITHUB_TOKEN"):
        repo_url = repo_url.replace("https://github.com/", f"https://{github_token}@github.com/") + ".git"

    agent.env.execute(f"git clone {repo_url} /testbed", cwd="/")

    exit_status, result = None, None
    try:
        exit_status, result = agent.run(task)
    except KeyboardInterrupt:
        console.print("\n[bold red]KeyboardInterrupt -- goodbye[/bold red]")
    finally:
        save_traj(agent, Path("traj.json"), exit_status=exit_status, result=result)
    return agent


if __name__ == "__main__":
    app()

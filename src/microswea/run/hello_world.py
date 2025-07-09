import os
from pathlib import Path

import typer
import yaml

from microswea import package_dir
from microswea.agents.default import DefaultAgent
from microswea.environments.local import LocalEnvironment
from microswea.models.litellm_model import LitellmModel

app = typer.Typer()


@app.command()
def main(
    problem: str = typer.Option(..., "-p", "--problem", help="Problem statement", show_default=False, prompt=True),
    model_name: str = typer.Option(
        os.getenv("MSWEA_MODEL_NAME"),
        "-m",
        "--model",
        help="Model name (defaults to MSWEA_MODEL_NAME env var)",
        prompt="What model do you want to use? (assuming that you've set the API key as the environment variable already)",
    ),
) -> DefaultAgent:
    agent = DefaultAgent(
        LitellmModel(model_name=model_name),
        LocalEnvironment(),
        **yaml.safe_load(Path(package_dir / "config" / "default.yaml").read_text())["agent"],
    )
    agent.run(problem)
    return agent


if __name__ == "__main__":
    app()

"""Run on a single SWE-Bench instance."""

from pathlib import Path

import typer
import yaml
from datasets import load_dataset

from minisweagent.agents.interactive import InteractiveAgent
from minisweagent.config import builtin_config_dir, get_config_path
from minisweagent.models import get_model
from minisweagent.run.extra.swebench import (
    DATASET_MAPPING,
    get_sb_environment,
)

app = typer.Typer(add_completion=False)


@app.command()
def main(
    subset: str = typer.Option("lite", "--subset", help="SWEBench subset to use or path to a dataset"),
    split: str = typer.Option("dev", "--split", help="Dataset split"),
    instance_spec: str = typer.Option(0, "-i", "--instance", help="SWE-Bench instance ID"),
    model_name: str | None = typer.Option(None, "-m", "--model", help="Model to use"),
    config_path: Path = typer.Option(
        builtin_config_dir / "extra" / "swebench.yaml", "-c", "--config", help="Path to a config file"
    ),
    environment_class: str | None = typer.Option(None, "--environment-class"),
) -> None:
    """Run on a single SWE-Bench instance."""
    dataset_path = DATASET_MAPPING.get(subset, subset)
    print(f"Loading dataset from {dataset_path}, split {split}...")
    instances = {
        inst["instance_id"]: inst  # type: ignore
        for inst in load_dataset(dataset_path, split=split)
    }
    if instance_spec.isnumeric():
        instance_spec = sorted(instances.keys())[int(instance_spec)]
    instance: dict = instances[instance_spec]  # type: ignore

    config = yaml.safe_load(get_config_path(config_path).read_text())
    config.setdefault("environment", {}).setdefault("environment_class", environment_class)
    env = get_sb_environment(config, instance)
    agent = InteractiveAgent(
        get_model(model_name, config.get("model", {})),
        env,
        **(config.get("agent", {}) | {"mode": "yolo"}),
    )
    agent.run(instance["problem_statement"])


if __name__ == "__main__":
    app()

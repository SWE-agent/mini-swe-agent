"""Run on a single SWE-Bench instance."""

from pathlib import Path

import typer
import yaml
from datasets import load_dataset

from microsweagent.agents.interactive import InteractiveAgent
from microsweagent.config import builtin_config_dir, get_config_path
from microsweagent.environments.docker import DockerEnvironment
from microsweagent.models import get_model
from microsweagent.run.extra.swebench import DATASET_MAPPING, get_swebench_docker_image_name

app = typer.Typer(add_completion=False)


@app.command()
def main(
    subset: str = typer.Option("lite", "--subset", help="SWEBench subset to use or path to a dataset"),
    split: str = typer.Option("dev", "--split", help="Dataset split"),
    instance_spec: str = typer.Option(None, "-i", "--instance", help="SWE-Bench instance ID"),
    model_name: str | None = typer.Option(None, "-m", "--model", help="Model to use"),
    config_path: Path = typer.Option(
        builtin_config_dir / "extra" / "swebench.yaml", "-c", "--config", help="Path to a config file"
    ),
) -> None:
    """Run on a single SWE-Bench instance."""
    try:
        dataset_path = DATASET_MAPPING[subset]
    except KeyError:
        dataset_path = subset
    print(f"Loading dataset {dataset_path}, split {split}...")
    instances = {
        inst["instance_id"]: inst  # type: ignore
        for inst in load_dataset(dataset_path, split=split)
    }
    if instance_spec.isnumeric():
        instance_spec = sorted(instances.keys())[int(instance_spec)]
    instance: dict = instances[instance_spec]  # type: ignore

    _config = yaml.safe_load(get_config_path(config_path).read_text())
    env = DockerEnvironment(**(_config.get("environment", {}) | {"image": get_swebench_docker_image_name(instance)}))
    agent = InteractiveAgent(
        get_model(model_name, _config.get("model", {})),
        env,
        **(_config.get("agent", {}) | {"mode": "yolo"}),
    )
    agent.run(instance["problem_statement"])


if __name__ == "__main__":
    app()

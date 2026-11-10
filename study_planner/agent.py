from pathlib import Path

import yaml
from jinja2 import Template

from study_planner.input import build_planner_context, load_planner_input
from study_planner.schemas import AvailabilityBlock, Task


CONFIG_PATH = Path("study_planner/config/study_planner.yaml")


def build_agent_prompt_from_data(
    tasks: list[Task],
    availability: list[AvailabilityBlock],
) -> str:
    context = build_planner_context(
        tasks=tasks,
        availability=availability,
    )

    with CONFIG_PATH.open() as file:
        config = yaml.safe_load(file)

    template = Template(
        config["agent"]["instance_template"]
    )

    return template.render(**context)


def build_agent_prompt(
    input_path: str | Path,
) -> str:
    tasks, availability = load_planner_input(
        input_path
    )

    return build_agent_prompt_from_data(
        tasks,
        availability,
    )
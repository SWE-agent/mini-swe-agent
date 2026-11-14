import json
import re
from datetime import datetime
from pathlib import Path

import typer

from study_planner.input import parse_planner_input

app = typer.Typer()


def _prompt_datetime(label: str) -> datetime:
    while True:
        value = typer.prompt(f"{label} (YYYY-MM-DD HH:MM)")
        try:
            return datetime.strptime(value, "%Y-%m-%d %H:%M")
        except ValueError:
            typer.echo("Enter a valid date and time in YYYY-MM-DD HH:MM format.", err=True)


def _prompt_positive_int(label: str) -> int:
    while True:
        value = typer.prompt(label)
        try:
            result = int(value)
        except ValueError:
            result = 0
        if result > 0:
            return result
        typer.echo("Enter a whole number greater than zero.", err=True)


def _task_id(name: str, index: int) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")
    return f"{slug or 'task'}_{index}"


def _collect_tasks() -> list[dict[str, object]]:
    tasks = []
    for index in range(1, _prompt_positive_int("How many tasks?") + 1):
        name = typer.prompt(f"Task {index} name").strip()
        while not name:
            typer.echo("Task name cannot be empty.", err=True)
            name = typer.prompt(f"Task {index} name").strip()
        course = typer.prompt(f"Task {index} course/category (optional)", default="").strip()
        deadline = _prompt_datetime(f"Task {index} deadline")
        estimated_minutes = _prompt_positive_int(f"Task {index} estimated workload (minutes)")
        priority = typer.prompt(f"Task {index} priority (1=low, 2=medium, 3=high)", type=int)
        while priority not in (1, 2, 3):
            typer.echo("Priority must be 1 (low), 2 (medium), or 3 (high).", err=True)
            priority = typer.prompt(f"Task {index} priority (1=low, 2=medium, 3=high)", type=int)
        tasks.append(
            {
                "id": _task_id(name, index),
                "name": name,
                "course": course or None,
                "deadline": deadline.isoformat(),
                "estimated_minutes": estimated_minutes,
                "priority": priority,
            }
        )
    return tasks


def _collect_availability() -> list[dict[str, str]]:
    blocks = []
    for index in range(1, _prompt_positive_int("How many availability windows?") + 1):
        while True:
            start = _prompt_datetime(f"Window {index} start")
            end = _prompt_datetime(f"Window {index} end")
            if end > start:
                break
            typer.echo("Window end must be after its start.", err=True)
        blocks.append({"start": start.isoformat(), "end": end.isoformat()})
    return blocks


@app.command()
def main(
    output: Path = typer.Option(
        Path("study_planner/data/planner_input.json"),
        "--output",
        "-o",
        help="Path for the validated planner input JSON.",
    ),
) -> None:
    """Collect tasks and availability for the study planner."""
    submitted = {"tasks": _collect_tasks(), "availability": _collect_availability()}
    try:
        tasks, availability = parse_planner_input(submitted)
    except ValueError as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(code=1) from exc

    data = {
        "tasks": [task.model_dump(mode="json") for task in tasks],
        "availability": [block.model_dump(mode="json") for block in availability],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(data, indent=2) + "\n")
    typer.echo(f"Validated planner input saved to {output}")


if __name__ == "__main__":
    app()

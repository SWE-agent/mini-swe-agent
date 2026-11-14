import json

from typer.testing import CliRunner

from study_planner.input import load_planner_input
from study_planner.ui.app import app


def test_interactive_input_is_validated_and_saved_for_pipeline(tmp_path):
    output = tmp_path / "nested" / "planner.json"
    result = CliRunner().invoke(
        app,
        ["--output", str(output)],
        input=("1\nStudy for final\nCS 179\n2026-10-28 23:59\n120\n3\n1\n2026-10-26 14:00\n2026-10-26 16:00\n"),
    )

    assert result.exit_code == 0, result.output
    assert "Validated planner input saved" in result.output
    tasks, availability = load_planner_input(output)
    assert (tasks[0].id, tasks[0].name, tasks[0].course, tasks[0].estimated_minutes, tasks[0].priority) == (
        "study_for_final_1",
        "Study for final",
        "CS 179",
        120,
        3,
    )
    assert availability[0].start.isoformat() == "2026-10-26T14:00:00"
    assert json.loads(output.read_text())["tasks"][0]["deadline"] == "2026-10-28T23:59:00"


def test_interactive_input_reprompts_for_invalid_workload(tmp_path):
    result = CliRunner().invoke(
        app,
        ["--output", str(tmp_path / "planner.json")],
        input=("1\nRead chapter\n\n2026-10-28 23:59\n0\n45\n2\n1\n2026-10-26 14:00\n2026-10-26 15:00\n"),
    )

    assert result.exit_code == 0, result.output
    assert "Enter a whole number greater than zero." in result.output
    assert load_planner_input(tmp_path / "planner.json")[0][0].estimated_minutes == 45

from study_planner.agent import build_agent_prompt
from study_planner.input import build_planner_context, load_planner_input


def test_load_planner_input():
    tasks, availability = load_planner_input(
        "eval_cases/simple_week.json"
    )

    assert len(tasks) == 1
    assert tasks[0].id == "cs_midterm"

    assert len(availability) == 1


def test_build_planner_context():
    tasks, availability = load_planner_input(
        "eval_cases/simple_week.json"
    )

    context = build_planner_context(tasks, availability)

    assert "cs_midterm" in context["tasks"]
    assert "2026-10-26T14:00:00" in context["availability"]


def test_build_agent_prompt():
    prompt = build_agent_prompt(
        "eval_cases/simple_week.json"
    )

    assert "Study for CS Midterm" in prompt
    assert "2026-10-26T14:00:00" in prompt
    assert "Deadline validation" in prompt
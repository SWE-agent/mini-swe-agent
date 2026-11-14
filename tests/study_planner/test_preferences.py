import pytest

from study_planner.agent import build_agent_prompt, build_agent_prompt_from_data
from study_planner.input import load_planner_input, parse_preferences
from study_planner.schemas import SchedulingPreferences


@pytest.mark.parametrize("raw", [{"study_period": "night"}, {"session_minutes": 0}, {"unknown": 1}])
def test_parse_preferences(raw):
    assert parse_preferences({}) == SchedulingPreferences()
    with pytest.raises(ValueError, match="Invalid scheduling preferences"):
        parse_preferences({"preferences": raw})


def test_prompt_includes_preferences():
    tasks, availability = load_planner_input("eval_cases/simple_week.json")
    assert '"study_period": "any"' in build_agent_prompt("eval_cases/simple_week.json")
    assert '"study_period": "evening"' in build_agent_prompt_from_data(
        tasks, availability, SchedulingPreferences(study_period="evening")
    )

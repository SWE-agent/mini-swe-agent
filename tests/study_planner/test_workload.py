import json
from pathlib import Path

import pytest

from study_planner.input import load_planner_input
from study_planner.tools.workload import validate_workload


@pytest.mark.parametrize(("case",), [(p,) for p in sorted(Path("eval_cases").glob("*.json"))])
def test_validate_workload_matches_eval_cases(case):
    result = validate_workload(*load_planner_input(case))
    assert result.valid == json.loads(case.read_text())["expected_output"]["feasible"]


def test_validate_workload_overloaded_week():
    issues = validate_workload(*load_planner_input("eval_cases/overloaded_week.json")).issues
    assert [i.message for i in issues] == [
        "Infeasible: 480 minutes of work (cs_midterm, math_midterm) due by 2026-10-27T17:00:00, "
        "but only 300 minutes available before then"
    ]

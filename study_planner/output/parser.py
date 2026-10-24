import json
from pydantic import ValidationError
from study_planner.schemas import Schedule

def parse_schedule(output: str) -> Schedule:
    try:
        # checks if valid JSON
        data = json.loads(output)
        # validates against the Schedule schema
        return Schedule.model_validate(data)
    except (json.JSONDecodeError, ValidationError) as e:
        raise ValueError(f"Invalid schedule output: {e}")
    
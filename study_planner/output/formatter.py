from study_planner.schemas import Schedule

def format_schedule(schedule: Schedule) -> str:
    return schedule.model_dump_json(indent=2)

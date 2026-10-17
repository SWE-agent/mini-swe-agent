# study_planner/schemas/schedule.py

from datetime import datetime

from pydantic import BaseModel


class StudyBlock(BaseModel):
    task_id: str
    start: datetime
    end: datetime


class Schedule(BaseModel):
    blocks: list[StudyBlock]

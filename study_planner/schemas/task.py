# study_planner/schemas/task.py

from datetime import datetime

from pydantic import BaseModel


class Task(BaseModel):
    id: str
    name: str
    course: str | None = None
    deadline: datetime
    estimated_minutes: int
    priority: int

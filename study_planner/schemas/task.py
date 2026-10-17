from datetime import datetime

from pydantic import BaseModel, Field


class Task(BaseModel):
    id: str
    name: str
    course: str | None = None
    deadline: datetime
    estimated_minutes: int = Field(gt=0)
    priority: int = Field(ge=1, le=3)
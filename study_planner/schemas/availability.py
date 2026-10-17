# study_planner/schemas/availability.py

from datetime import datetime
from pydantic import BaseModel


class AvailabilityBlock(BaseModel):
    start: datetime
    end: datetime
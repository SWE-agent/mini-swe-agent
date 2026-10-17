from datetime import datetime

from pydantic import BaseModel, model_validator


class AvailabilityBlock(BaseModel):
    start: datetime
    end: datetime

    @model_validator(mode="after")
    def validate_time_range(self):
        if self.end <= self.start:
            raise ValueError("end must be after start")

        return self
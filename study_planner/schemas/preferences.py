from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class SchedulingPreferences(BaseModel):
    model_config = ConfigDict(extra="forbid")

    study_period: Literal["morning", "afternoon", "evening", "any"] = "any"
    session_minutes: int | None = Field(default=None, gt=0)
    min_break_minutes: int = Field(default=0, ge=0)
    spread_across_days: bool = False

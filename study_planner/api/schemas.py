from typing import Literal

from pydantic import BaseModel

from study_planner.schemas import (
    AvailabilityBlock,
    Schedule,
    Task,
)


class PlanRequest(BaseModel):
    tasks: list[Task]
    availability: list[AvailabilityBlock]


class PlanSuccessResponse(BaseModel):
    status: Literal["success"] = "success"
    schedule: Schedule


class PlanInfeasibleResponse(BaseModel):
    status: Literal["infeasible"] = "infeasible"
    message: str


class PlanErrorResponse(BaseModel):
    status: Literal["error"] = "error"
    message: str

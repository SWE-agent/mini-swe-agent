from fastapi import FastAPI
from fastapi.responses import JSONResponse

from study_planner.api.schemas import (
    PlanErrorResponse,
    PlanInfeasibleResponse,
    PlanRequest,
    PlanSuccessResponse,
)
from study_planner.loop import (
    run_planning_loop_from_data,
)
from study_planner.model import generate_schedule


app = FastAPI(
    title="Study Planner API",
    version="0.1.0",
)


@app.get("/health")
def health():
    return {
        "status": "ok",
    }


@app.post(
    "/api/plan",
    response_model=PlanSuccessResponse,
    responses={
        422: {
            "model": PlanInfeasibleResponse,
        },
        500: {
            "model": PlanErrorResponse,
        },
    },
)
def create_plan(
    request: PlanRequest,
):
    try:
        schedule = run_planning_loop_from_data(
            request.tasks,
            request.availability,
            generate_schedule,
        )

        return PlanSuccessResponse(
            schedule=schedule
        )

    except ValueError as exc:
        message = str(exc)

        if "infeasible" in message.lower():
            return JSONResponse(
                status_code=422,
                content=PlanInfeasibleResponse(
                    message=message
                ).model_dump(mode="json"),
            )

        return JSONResponse(
            status_code=400,
            content=PlanErrorResponse(
                message=message
            ).model_dump(mode="json"),
        )

    except Exception:
        return JSONResponse(
            status_code=500,
            content=PlanErrorResponse(
                message=(
                    "Unable to generate study plan."
                )
            ).model_dump(mode="json"),
        )
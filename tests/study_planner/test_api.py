import importlib
import json

from fastapi.testclient import TestClient

from study_planner.api.app import app


app_module = importlib.import_module(
    "study_planner.api.app"
)

client = TestClient(app)


VALID_SCHEDULE = json.dumps(
    {
        "blocks": [
            {
                "task_id": "cs_midterm",
                "start": "2026-10-26T14:00:00",
                "end": "2026-10-26T16:00:00",
            }
        ]
    }
)


SIMPLE_REQUEST = {
    "tasks": [
        {
            "id": "cs_midterm",
            "name": "Study for CS Midterm",
            "course": "CS100",
            "deadline": (
                "2026-10-28T23:59:00"
            ),
            "estimated_minutes": 120,
            "priority": 2,
        }
    ],
    "availability": [
        {
            "start": (
                "2026-10-26T14:00:00"
            ),
            "end": (
                "2026-10-26T16:00:00"
            ),
        }
    ],
}


def test_health_endpoint():
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {
        "status": "ok",
    }


def test_plan_endpoint_returns_schedule(
    monkeypatch,
):
    monkeypatch.setattr(
        app_module,
        "generate_schedule",
        lambda prompt: VALID_SCHEDULE,
    )

    response = client.post(
        "/api/plan",
        json=SIMPLE_REQUEST,
    )

    assert response.status_code == 200

    body = response.json()

    assert body["status"] == "success"

    assert (
        body["schedule"]["blocks"][0]["task_id"]
        == "cs_midterm"
    )

def test_plan_endpoint_reports_infeasible(
    monkeypatch,
):
    monkeypatch.setattr(
      app_module,
      "generate_schedule",
      lambda prompt: VALID_SCHEDULE,
    )

    request = {
        "tasks": [
            {
                "id": "cs_midterm",
                "name": "CS Midterm",
                "deadline": (
                    "2026-10-27T17:00:00"
                ),
                "estimated_minutes": 240,
                "priority": 1,
            },
            {
                "id": "math_midterm",
                "name": "Math Midterm",
                "deadline": (
                    "2026-10-27T17:00:00"
                ),
                "estimated_minutes": 240,
                "priority": 2,
            },
        ],
        "availability": [
            {
                "start": (
                    "2026-10-26T09:00:00"
                ),
                "end": (
                    "2026-10-26T12:00:00"
                ),
            },
            {
                "start": (
                    "2026-10-27T09:00:00"
                ),
                "end": (
                    "2026-10-27T11:00:00"
                ),
            },
        ],
    }

    response = client.post(
        "/api/plan",
        json=request,
    )

    assert response.status_code == 422

    body = response.json()

    assert body["status"] == "infeasible"
    assert "480 minutes" in body["message"]
    assert "300 minutes" in body["message"]

def test_plan_endpoint_rejects_invalid_task():
    response = client.post(
        "/api/plan",
        json={
            "tasks": [
                {
                    "id": "bad-task",
                }
            ],
            "availability": [],
        },
    )

    assert response.status_code == 422
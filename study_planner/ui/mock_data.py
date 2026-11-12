from study_planner.schemas import Schedule

mock_schedule = Schedule.model_validate({
    "blocks": [
        {
            "task_id": "cs_midterm",
            "start": "2026-11-16T09:00:00",
            "end": "2026-11-16T11:00:00"
        },
        {
            "task_id": "math_midterm",
            "start": "2026-11-16T13:00:00",
            "end": "2026-11-16T14:30:00"
        },
        {
            "task_id": "cs_midterm",
            "start": "2026-11-17T10:00:00",
            "end": "2026-11-17T11:00:00"
        },
        {
            "task_id": "biology_quiz",
            "start": "2026-11-18T14:00:00",
            "end": "2026-11-18T16:00:00"
        },
        {
            "task_id": "math_midterm",
            "start": "2026-11-19T09:00:00",
            "end": "2026-11-19T10:30:00"
        }
    ]
})


mock_tasks = {
    "cs_midterm": {
        "name": "Study for CS Midterm",
        "course": "CS 166"
    },
    "math_midterm": {
        "name": "Study for Math Midterm",
        "course": "MATH 031"
    },
    "biology_quiz": {
        "name": "Prepare for Biology Quiz",
        "course": "BIOL 005"
    }
}

empty_schedule = Schedule(blocks=[])

overnight_schedule = Schedule.model_validate({
    "blocks": [
        {
            "task_id": "cs_midterm",
            "start": "2026-11-16T23:00:00",
            "end": "2026-11-17T01:00:00"
        }
    ]
})

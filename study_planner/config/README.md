# Study Planner Agent Configuration

This directory contains the study-planner-specific agent instructions used with
mini-SWE-agent.

## study_planner.yaml

The configuration defines the behavior expected from the planning agent.

The agent receives:

- Tasks using the shared `Task` schema
- Availability using the shared `AvailabilityBlock` schema
- Optional scheduling preferences using the `SchedulingPreferences` schema

The agent produces a schedule matching the shared `Schedule` schema:

```json
{
  "blocks": [
    {
      "task_id": "cs_project",
      "start": "2026-10-26T18:00:00",
      "end": "2026-10-26T19:30:00"
    }
  ]
}
```

## Scheduling preferences

The planner input may include an optional `preferences` object. Inputs without
it, or with only some fields, use the defaults below.

```json
{
  "tasks": [],
  "availability": [],
  "preferences": {
    "study_period": "evening",
    "session_minutes": 60,
    "min_break_minutes": 15,
    "spread_across_days": true
  }
}
```

| Field | Values | Default | Expected behavior |
|---|---|---|---|
| `study_period` | `morning` (before 12:00), `afternoon` (12:00-17:00), `evening` (after 17:00), `any` | `any` | Prefer blocks inside this period |
| `session_minutes` | positive integer or `null` | `null` | Prefer blocks of about this length |
| `min_break_minutes` | integer >= 0 | `0` | Leave at least this gap between consecutive blocks when possible |
| `spread_across_days` | `true` / `false` | `false` | Prefer spreading a task's work over several days |

Preferences are soft. The agent uses them only to choose between schedules that
already pass every validator, and never breaks a deadline, availability,
completion, or overlap rule to satisfy one. They are not checked by the
deterministic validators.
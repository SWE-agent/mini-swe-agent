# Study Planner Agent Configuration

This directory contains the study-planner-specific agent instructions used with
mini-SWE-agent.

## study_planner.yaml

The configuration defines the behavior expected from the planning agent.

The agent receives:

- Tasks using the shared `Task` schema
- Availability using the shared `AvailabilityBlock` schema

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
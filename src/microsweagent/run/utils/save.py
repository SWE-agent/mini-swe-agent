import json
from pathlib import Path

from microsweagent import Agent


def save_traj(agent: Agent | None, path: Path, *, exit_status: str | None = None, result: str | None = None, **kwargs):
    if agent is not None:
        data = {
            "info": {
                "exit_status": exit_status,
                "submission": result,
                "model_stats": {
                    "instance_cost": agent.model.cost,
                    "api_calls": agent.model.n_calls,
                },
            },
            "messages": agent.messages,
        } | kwargs
    else:
        # Create minimal trajectory when agent is None
        data = {
            "info": {
                "exit_status": exit_status,
                "submission": result,
                "model_stats": {
                    "instance_cost": 0.0,
                    "api_calls": 0,
                },
            },
            "messages": [],
        } | kwargs

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2))

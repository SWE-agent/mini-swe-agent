import json
from pathlib import Path

from microsweagent import Agent


def save_traj(agent: Agent, path: Path, *, exit_status: str | None = None, result: str | None = None, **kwargs):
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
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2))

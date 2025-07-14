import json
from pathlib import Path

from microsweagent import Agent


def save_traj(
    agent: Agent | None,
    path: Path,
    *,
    exit_status: str | None = None,
    result: str | None = None,
    extra_info: dict | None = None,
    **kwargs,
):
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
    if agent is not None:
        data["info"]["model_stats"]["instance_cost"] = agent.model.cost
        data["info"]["model_stats"]["api_calls"] = agent.model.n_calls
        data["messages"] = agent.messages
    if extra_info:
        data["info"].update(extra_info)

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2))

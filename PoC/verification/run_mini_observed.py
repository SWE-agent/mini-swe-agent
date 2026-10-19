"""Run mini-swe-agent's DefaultAgent once with Claude, recording observatory_log events for the kit dashboard.

Same setup as examples/agent_dashboard: config mini.yaml, local environment in a fresh workspace folder, cost tracking
errors ignored (LiteLLM may not know the price of a new model).

usage (repo root):
  OBSERVATORY_RUNS=observatory/runs uv run --env-file .env python PoC/verification/run_mini_observed.py "<task>" [model] [steps]
The bash commands run on this machine (inside the workspace folder, not a sandbox).
"""

import sys
import time
from pathlib import Path

from minisweagent.agents.default import DefaultAgent
from minisweagent.config import get_config_from_spec
from minisweagent.environments import get_environment
from minisweagent.models import get_model

task = sys.argv[1]
model_name = sys.argv[2] if len(sys.argv) > 2 else "anthropic/claude-haiku-5-5"
steps = int(sys.argv[3]) if len(sys.argv) > 3 else 8

config = get_config_from_spec("mini.yaml")
workspace = Path(__file__).parent / "mini_workspaces" / time.strftime("%Y%m%d-%H%M%S")
workspace.mkdir(parents=True)
agent = DefaultAgent(
    get_model(model_name, config["model"] | {"cost_tracking": "ignore_errors"}),
    get_environment(config["environment"] | {"cwd": str(workspace)}, default_type="local"),
    **{k: v for k, v in config["agent"].items() if k != "mode"} | {"step_limit": steps},
)
info = agent.run(task)
print(f"workspace: {workspace}\nexit_status: {info.get('exit_status')}  steps: {agent.n_calls}  cost: ${agent.cost:.4f}")
print(f"submission: {info.get('submission', '')[:500]}")

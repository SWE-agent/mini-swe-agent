"""Check run script for Agent Observatory logging."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from minisweagent.agents.default import DefaultAgent
from minisweagent.environments.local import LocalEnvironment
from minisweagent.models.test_models import DeterministicModel, make_output

outputs = [
    make_output(content="First step, let's run a command.", actions=[{"command": "echo hello"}]),
    make_output(content="Task complete, submit.", actions=[{"command": "echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT"}]),
]

model = DeterministicModel(outputs=outputs)
env = LocalEnvironment()
config = {
    "system_template": "You are a helpful assistant.",
    "instance_template": "Task: {{task}}",
    "step_limit": 10,
    "cost_limit": 10.0,
}
agent = DefaultAgent(model=model, env=env, **config)
result = agent.run(task="test task")
print("Agent finished with:", result)

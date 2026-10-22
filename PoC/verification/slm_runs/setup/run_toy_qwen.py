"""Run the toy agent from the current directory with Qwen as its model, without changing the agent's code."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path.cwd()))
from agent import ToyAgent

agent = ToyAgent()
agent.llm.model = "Qwen/Qwen3.8-27B"
print(agent.run(" ".join(sys.argv[1:]) or input("Task: ")))

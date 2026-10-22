import os
os.environ["OBSERVATORY_RUNS"] = "observatory/runs"
import sys
sys.path.insert(0, ".")
from agent import ToyAgent

agent = ToyAgent()
result = agent.run("what is 2+2", max_steps=3)
print("Result:", result)
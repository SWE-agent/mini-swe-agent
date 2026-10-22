import os
import sys

# Set the observatory runs env var
os.environ["OBSERVATORY_RUNS"] = r"observatory\runs"

# Add the current directory to path
sys.path.insert(0, "/tmp/agent_path")

# Import and run the agent
from agent import ToyAgent

agent = ToyAgent()
result = agent.run("what is 2+2")
print("Result:", result)

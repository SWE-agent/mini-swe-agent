import os
os.environ["OBSERVATORY_RUNS"] = "/tmp/obs_verify_runs"
import sys
sys.path.insert(0, ".")
from agent import ToyAgent
result = ToyAgent().run("say hello")
print(f"Result: {result}")

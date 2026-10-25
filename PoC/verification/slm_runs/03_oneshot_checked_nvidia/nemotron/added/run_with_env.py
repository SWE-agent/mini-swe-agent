#!/usr/bin/env python
import os

os.environ["OBSERVATORY_RUNS"] = "observatory/runs"
import sys

sys.path.insert(
    0,
    "/tmp/test_claude/c--Users-USER-Developments-mini-swe-agent/2b6776a6-acde-42c1-9433-2f8ca7ccb7f5/scratchpad/slm/v2/nemotron",
)
from agent import ToyAgent

agent = ToyAgent()
result = agent.run("what is 2+2")
print("Result:", result)

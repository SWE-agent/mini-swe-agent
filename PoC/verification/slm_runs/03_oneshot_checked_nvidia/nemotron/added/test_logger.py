import os
import sys

# Set OBSERVATORY_RUNS before importing anything from the agent
os.environ["OBSERVATORY_RUNS"] = "observatory/runs"

# Add the agent package to path
sys.path.insert(0, "/tmp/test_path")

import observatory_log

observatory_log._files.clear()
observatory_log.llm_query(1, model="test", messages=[])
observatory_log.llm_response(
    1, actions=[], reply={"content": "hi", "calls": [], "message": {"role": "assistant", "content": "hi"}}
)
s = observatory_log.tool_start("bash", {"command": "ls"})
observatory_log.tool_end(s, "bash", result="listing")
print("logged OK")

import os
import sys

# Ensure observatory_log is importable
sys.path.insert(0, os.path.dirname(__file__))

# Set OBSERVATORY_RUNS before importing observatory_log
os.environ["OBSERVATORY_RUNS"] = r"observatory\runs"

# Now import
import observatory_log

# Simple test
observatory_log._files.clear()
observatory_log.llm_query(1, model="test-model", messages=[{"role": "user", "content": "test"}])
observatory_log.llm_response(1, actions=[], reply={"content": "test response", "calls": [], "message": {"role": "assistant", "content": "test response"}}, messages=[{"role": "user", "content": "test"}])
start = observatory_log.tool_start("bash", {"command": "echo hello"})
observatory_log.tool_end(start, "bash", result="hello")
observatory_log.tool_error(start, "bash", RuntimeError("boom"))

print("Test passed - check observatory/runs/ for JSONL file")
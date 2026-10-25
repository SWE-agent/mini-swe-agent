import json
import os
import tempfile
from pathlib import Path

os.environ["OBSERVATORY_RUNS"] = str(Path(tempfile.mkdtemp()) / "runs")

import observatory_log as log

# Clear cached files
log._files.clear()

# Test: logger is silent without the variable
os.environ.pop("OBSERVATORY_RUNS", None)
log.llm_query(1, model="m", messages=2)
runs = Path(os.environ.get("OBSERVATORY_RUNS", ""))
if runs.exists():
    lines = runs.read_text().splitlines()
    print(f"FAIL: expected no files, got {len(lines)} lines")
else:
    print("PASS: logger silent without OBSERVATORY_RUNS")

# Test: logger writes one line per event
os.environ["OBSERVATORY_RUNS"] = str(Path(tempfile.mkdtemp()) / "runs")
log._files.clear()
log.llm_query(1, model="m", messages=2, purpose="action")
log.llm_response(1, actions=[{"bash": {"command": "ls"}}], cost=0.01)
start = log.tool_start("bash", {"command": "ls"})
log.tool_end(start, "bash", result="exit 0")
log.llm_error(2, ValueError("bad"))
(file,) = Path(os.environ["OBSERVATORY_RUNS"]).glob("*.jsonl")
events = [json.loads(l) for l in file.read_text().splitlines()]
expected = [
    ("llm_query", 1),
    ("llm_response", 1),
    ("tool_start", None),
    ("tool_end", None),
    ("llm_error", 2),
]
actual = [(e["kind"], e["step"]) for e in events]
print(f"Events: {actual}")
if actual == expected:
    print("PASS: one line per event")
else:
    print("FAIL: event sequence mismatch")

# Check data
if events[0]["data"] == {"model": "m", "messages": 2, "purpose": "action"}:
    print("PASS: llm_query data correct")
else:
    print(f"FAIL: llm_query data: {events[0]['data']}")

if events[1]["data"]["actions"] == [{"bash": {"command": "ls"}}] and events[1]["data"].get("cost") == 0.01:
    print("PASS: llm_response data correct")
else:
    print(f"FAIL: llm_response data: {events[1]['data']}")

if events[3]["data"]["result"] == "exit 0" and events[3]["data"].get("duration_seconds", 0) >= 0:
    print("PASS: tool_end data correct")
else:
    print(f"FAIL: tool_end data: {events[3]['data']}")

if events[5]["data"]["error_type"] == "RuntimeError" and events[5]["data"]["error"] == "boom":
    print("PASS: llm_error data correct")
else:
    print(f"FAIL: llm_error data: {events[5]['data']}")

print("\nAll tests done.")

import json

data = {
    "logger": "observatory_log",
    "entry": {"function": "ToyAgent.run", "path": "agent.py", "line": 27, "inputs": ["task"]},
    "sources": [
        {"kind": "llm_response", "path": "agent.py", "line": 40, "call": "self.llm.chat", "label": "model call"},
        {"kind": "user_input", "path": "agent.py", "line": 59, "call": "input", "label": "prompt"},
    ],
}

with open("./observatory/diagram/verify-anchors.json", "w") as f:
    json.dump(data, f, indent=2)
    f.write("\n")

print("Done")

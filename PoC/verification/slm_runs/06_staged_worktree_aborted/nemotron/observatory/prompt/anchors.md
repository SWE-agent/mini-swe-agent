## This step: find the anchors (read the code, change no code)

Write `{kit}/diagram/verify-anchors.json`: where the agent's run starts, and the calls whose return value is an LLM reply or text a user typed. A static checker starts from these anchors to follow the values through the code.

```json
{
  "logger": "observatory_log",
  "entry": {"function": "Agent.run", "path": "pkg/agent.py", "line": 12, "inputs": ["task"]},
  "sources": [
    {"kind": "llm_response", "path": "pkg/llm.py", "line": 30, "call": "self.client.messages.create", "label": "model call"},
    {"kind": "user_input", "path": "pkg/agent.py", "line": 58, "call": "input", "label": "confirm prompt"}
  ]
}
```

- `entry`: the function that runs one task (`Class.method` or `function`), its file, the line of its `def`, and its parameters that carry the user's task.
- `sources`, only two kinds, one entry per call site:
  - `llm_response`: every call that returns the model's reply. Take the innermost call in this repo: the SDK call (`self.client.messages.create`, `litellm.completion`), or, when the model is an object passed in whose code is elsewhere, the call on it (`self.model.query`). Include fallback and retry calls.
  - `user_input`: every call inside a function that returns text the user types while the agent runs (`input`, `prompt_session.prompt`). Not calls at module level.
- `call`: the called expression exactly as written in the source, without arguments (`self.client.messages.create`, `input`). `line`: the line where that call starts.
- Do not list tool calls: the checker finds them between two LLM calls.

## This step: find the anchors (read the code, change no code)

{scope}Write `{kit}/diagram/verify-anchors.json`: where the agent's run starts, and the calls whose return value is an LLM reply or text a user typed. After the logging is added in the next steps, a static checker starts from these anchors to verify that it records those values; the lines may move then, the checker finds the calls again.

```json
{
  "logger": "observatory_log",
  "entry": {"function": "Agent.run", "path": "pkg/agent.py", "line": 12, "inputs": ["task"]},
  "sources": [
    {"kind": "llm_response", "path": "pkg/agent.py", "line": 30, "call": "self.model.query", "label": "model call"},
    {"kind": "user_input", "path": "pkg/agent.py", "line": 58, "call": "input", "label": "confirm prompt"}
  ]
}
```

- `entry`: the function that runs one task (`Class.method` or `function`), its file, the line of its `def`, and its parameters that carry the user's task.
- `sources`, only two kinds, one entry per call site:
  - `llm_response`: every call that returns the model's reply to the agent. Which call depends on how the agent gets its model:
    - The agent receives the model as an object (a constructor parameter, a factory, a config), so the code does not show which class it is: take the call on that object in the agent's code, for example `self.model.query`, even when the model classes are in this repo.
    - The agent creates the model client itself, directly (`self.client = anthropic.Anthropic()`) or through a wrapper class of this repo (`self.llm = LLM(...)`): take the SDK call that the wrapper makes (`self.client.messages.create`, `litellm.completion`), not the call on the wrapper (`self.llm.chat`).
    Include fallback and retry calls.
  - `user_input`: every call inside a function that returns text the user types while the agent runs (`input`, `prompt_session.prompt`). Not calls at module level.
- `call`: the called expression exactly as written in the source, without arguments (`self.client.messages.create`, `input`). `line`: the line where that call starts.
- Do not list tool calls: the checker finds them between two LLM calls.

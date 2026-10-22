## This step: the architecture diagram (left panel of the dashboard)

Write `{kit}/diagram/agent-architecture.json`, an Archify architecture spec of this agent's loop. It is shown next to the history so that a reader can follow each logged event to the code that reported it. The kit validates the file with Archify and generates the HTML; it sends you the errors if there are any. Read only `{kit}/verify/graph.json` and the code it points to: Archify's own documentation is not needed, its format is below.

### The input: `{kit}/verify/graph.json`

The data flow a static checker read from the code. `nodes`: each is a function (`role` entry or function), a field of the agent's state (`state`), or a call that leaves the agent (`llm_response`, `tool_response`, `user_input`, `final_output`), with its `path`, `line` and `end_line`. A node's `logs` are the logger calls in it: the `event`, `at` (the `path:line` that every logged event on the dashboard carries) and the values it `records`. `gaps` lists values that pass the node without being logged. `edges`: a value moving `from` one node `to` another, with the variable names in `values`.

### The output

```json
{
  "schema_version": 1,
  "diagram_type": "architecture",
  "meta": {
    "title": "Toy Agent Loop",
    "output": "static/agent-architecture.html",
    "quality_profile": "showcase",
    "repository": {"url": "https://github.com/owner/repo", "provider": "github", "link_mode": "web", "revision": "0000000000000000000000000000000000000000"}
  },
  "layout": {"mode": "grid", "cols": 5},
  "components": [
    {"id": "task", "type": "external", "label": "task", "sublabel": "user input", "row": 1, "col": 0},
    {"id": "run", "type": "backend", "label": "Agent.run", "sublabel": "step loop", "tag": "user_input", "row": 1, "col": 1,
     "sources": [{"path": "agent.py", "line": 26, "end_line": 33, "label": "run"}, {"path": "agent.py", "line": 27, "label": "tool_start user_input"}]},
    {"id": "chat", "type": "cloud", "label": "LLM.chat", "sublabel": "model call", "tag": "llm_query/response/error", "row": 1, "col": 2,
     "sources": [{"path": "llm.py", "line": 13, "end_line": 35, "label": "chat"}]}
  ],
  "boundaries": [{"kind": "region", "label": "one step", "wraps": ["run", "chat"], "pad": 24}],
  "connections": [
    {"from": "task", "to": "run", "label": "task"},
    {"from": "run", "to": "chat", "label": "messages"}
  ],
  "cards": [{"dot": "cyan", "title": "LLM boundary", "items": ["llm_query before every model call, llm_response with the whole reply"]}]
}
```

Rules:
- Components: one for each node that has `logs`, for each call that leaves the agent (`llm_response`, `tool_response`, `user_input`, `final_output`) and for the main `state` field; merge a small helper without logs into the function that calls it. At most 12. `id`: short snake_case. `label`: the node's real name. `sublabel`: a few words.
- `type`: `backend` for the agent's functions, `cloud` for the LLM call, `external` for tools, user input and the final output, `database` for state.
- `tag` (only on components with logs): the events, at most 24 characters, for example `llm_query/response/error`, `tool_start/end/error`, `user_input`.
- `sources`: 1 to 3 per component, from the graph: the node's `path`, `line`, `end_line`, and then the `at` lines of its logs (`path` and `line` split from `at`). Paths exactly as in the graph.
- Layout: `row` and `col` on a grid, one component per cell. Put the loop in reading order from left to right (task, run, LLM call, tools, back), state above it and user input or final output below it, so that every connection joins two cells next to each other where possible. Use 2 or 3 rows.
- `connections`: from the graph's `edges` between the components you drew, `label` one variable name. At most one connection per pair of components and direction.
- `cards`: 2 or 3, short: where each kind of event is logged, and any `gaps`.
- Keep `meta.repository` as shown: the kit sets its url and revision.

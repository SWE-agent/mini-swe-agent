This repo is an LLM agent. `{kit}/` contains the Agent Observatory: a dashboard (`{kit}/dashboard.py`) that shows a per-step history of an agent's LLM calls and tool executions, fed by a small logger module (`{kit}/observatory_log.py`). The dashboard and the logger are fixed and agent-agnostic: do not modify them. Your job is to connect this agent to the logger.

This session is non-interactive: nobody can answer questions, so never ask; carry out every part below to the end. If the agent already calls the logger, check that wiring against the steps below, fix only what is missing or wrong, and still write every file the steps ask for.

## Part 1: wire the logger into the agent

1. Copy `{kit}/observatory_log.py` unchanged into the agent's main Python package (the package that contains the agent loop) and import it from there. Read its docstring first: it lists the six functions and when to call each.
2. Find where the agent calls the model and where it executes a tool or action, and add the calls:
   - `llm_query` right before every model call, `llm_response` right after the reply was parsed into actions, `llm_error` when the call raises (then the agent's existing error handling continues unchanged).
   - `tool_start` right before one tool or action runs, `tool_end` after it returned, `tool_error` when it raised (then re-raise unchanged).
   - Report the user's task too: right where the agent receives it (its run function), and every place it reads user input, as the tool action `"user_input"` (`tool_start` with the prompt, `tool_end` with what the user typed).
   - Log whole values, never parts of them: a checker verifies that the user input, the model reply and the tool output each reach a logger call as the variable itself. Pass the variable (`messages=messages`, `response=response`, `result=output`), not a count (`len(messages)`), a slice (`output[:500]`), one field (`response.stop_reason`) or a value built from it (`[a["command"] for a in actions]`). Such summaries may be added as extra fields next to the whole value. For example:
     ```python
     observatory_log.llm_query(step, model=self.model, messages=messages)
     response = self.client.messages.create(model=self.model, messages=messages)
     reply = self.parse(response)
     observatory_log.llm_response(step, actions=[c["name"] for c in reply["calls"]], reply=reply, response=response)
     ...
     start = observatory_log.tool_start(name, args)
     out = self.tools[name](**args)
     observatory_log.tool_end(start, name, result=out)
     ```
   - Other useful keyword fields: the model name, a purpose, cost, token counts.
3. Rules:
   - Every code path that calls the model or executes an action must be covered, including subclasses and alternative agent classes that bypass the base-class path. If several paths run actions, wire the tool calls at the one place they all go through.
   - If the agent signals a normal outcome with an exception (for example finishing or submitting a task), report it with `tool_end` and a short result such as "submitted", not with `tool_error`.
   - No behavior change besides the logging. The agent's normal user-facing output must not change. Follow AGENTS.md if the repo has one (minimal, concise code, no extra comments).
   - Never put secrets the agent is configured to hide (API keys, sensitive data) into the fields.
4. Check your work: run the agent, or its smallest entry point, with a scripted or deterministic model so that no API key or network is needed. Start it with plain `uv run python <script>`: the environment variable `OBSERVATORY_RUNS` is already set to `{kit}/runs` for you, so do not put it in front of the command. Then read the JSON-lines file created in `{kit}/runs/` and confirm that every step shows its llm_query, llm_response and tool events in order. Leave that file there.
5. Write `{kit}/diagram/verify-anchors.json`, the anchors a static checker starts from to verify that your logging records the user input, the LLM response and the tool output before they leave the agent. All paths are relative to the repo root and all lines are the code as it is after your changes:
   ```json
   {
     "logger": "observatory_log",
     "entry": {"function": "Class.method", "path": "pkg/agent.py", "line": 1, "inputs": ["task"]},
     "sources": [
       {"kind": "llm_response", "path": "pkg/agent.py", "line": 1, "call": "self.model.query", "label": "model call"},
       {"kind": "tool_response", "path": "pkg/agent.py", "line": 1, "call": "self.env.execute", "label": "tool call"},
       {"kind": "user_input", "path": "pkg/agent.py", "line": 1, "call": "input", "label": "prompt"}
     ]
   }
   ```
   - `entry`: the agent's run function and its parameters that carry the user's task.
   - `sources`: every call whose return value is a model reply, a tool or action output, or text the user typed. `call` is the called expression exactly as written in the source (`ast.unparse` of the call's function) on that `line`.

<!-- diagram -->
## Part 2: the architecture diagram (left panel of the dashboard)

Write `{kit}/diagram/agent-architecture.json`, the Archify architecture spec of this agent's loop, which is shown next to the history. A separate tool generates the HTML from it, so never create or edit HTML.

- Read Archify's instructions at `{archify_dir}/SKILL.md` and the references it points to. Follow `{kit}/diagram/example-architecture.json`, the diagram of another agent, for the style and structure.
- Show the real flow of this agent: the task coming in, the agent loop, the LLM call, tool execution and the result going out, with this repo's real classes and functions as component labels.
- Every component that corresponds to code carries `sources`: the file `path` and `line` (and `end_line`) of that code in this repo as it is now, including your logger calls. Give the components that carry your logger calls a `tag` naming the events, for example "llm_query llm_response llm_error".
- Keep the `meta.repository` block; the kit sets its url and revision itself.
- Validate with `node {archify_dir}/bin/archify.mjs validate architecture {kit}/diagram/agent-architecture.json --quality showcase` until it passes. Add `--repo-root .` if the tool asks for it. Errors about the repository evidence (the revision of your uncommitted changes) can be ignored: the kit pins the revision itself afterwards and validates again.

Do not commit.

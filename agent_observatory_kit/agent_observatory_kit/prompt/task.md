This repo is an LLM agent. `{kit}/` contains the Agent Observatory: a dashboard (`{kit}/dashboard.py`) that shows a per-step history of an agent's LLM calls and tool executions, fed by a small logger module (`{kit}/observatory_log.py`). The dashboard and the logger are fixed and agent-agnostic: do not modify them. Your job is to connect this agent to the logger.

## Part 1: wire the logger into the agent

1. Copy `{kit}/observatory_log.py` unchanged into the agent's main Python package (the package that contains the agent loop) and import it from there. Read its docstring first: it lists the six functions and when to call each.
2. Find where the agent calls the model and where it executes a tool or action, and add the calls:
   - `llm_query` right before every model call, `llm_response` right after the reply was parsed into actions, `llm_error` when the call raises (then the agent's existing error handling continues unchanged).
   - `tool_start` right before one tool or action runs, `tool_end` after it returned, `tool_error` when it raised (then re-raise unchanged).
   - Pass what helps a reader as keyword fields: the model name, the number of messages, a purpose, cost, token counts, a short result.
3. Rules:
   - Every code path that calls the model or executes an action must be covered, including subclasses and alternative agent classes that bypass the base-class path. If several paths run actions, wire the tool calls at the one place they all go through.
   - If the agent signals a normal outcome with an exception (for example finishing or submitting a task), report it with `tool_end` and a short result such as "submitted", not with `tool_error`.
   - No behavior change besides the logging. The agent's normal user-facing output must not change. Follow AGENTS.md if the repo has one (minimal, concise code, no extra comments).
   - Never put secrets the agent is configured to hide (API keys, sensitive data) into the fields.
4. Check your work: run the agent, or its smallest entry point, with a scripted or deterministic model so that no API key or network is needed. Start it with plain `uv run python <script>`: the environment variable `OBSERVATORY_RUNS` is already set to `{kit}/runs` for you, so do not put it in front of the command. Then read the JSON-lines file created in `{kit}/runs/` and confirm that every step shows its llm_query, llm_response and tool events in order. Leave that file there.

<!-- diagram -->
## Part 2: the architecture diagram (left panel of the dashboard)

Write `{kit}/diagram/agent-architecture.json`, the Archify architecture spec of this agent's loop, which is shown next to the history. A separate tool generates the HTML from it, so never create or edit HTML.

- Read Archify's instructions at `{archify_dir}/SKILL.md` and the references it points to. Follow `{kit}/diagram/example-architecture.json`, the diagram of another agent, for the style and structure.
- Show the real flow of this agent: the task coming in, the agent loop, the LLM call, tool execution and the result going out, with this repo's real classes and functions as component labels.
- Every component that corresponds to code carries `sources`: the file `path` and `line` (and `end_line`) of that code in this repo as it is now, including your logger calls. Give the components that carry your logger calls a `tag` naming the events, for example "llm_query llm_response llm_error".
- Keep the `meta.repository` block; the kit sets its url and revision itself.
- Validate with `node {archify_dir}/bin/archify.mjs validate architecture {kit}/diagram/agent-architecture.json --quality showcase` until it passes. Add `--repo-root .` if the tool asks for it. Errors about the repository evidence (the revision of your uncommitted changes) can be ignored: the kit pins the revision itself afterwards and validates again.

Do not commit.

# Agent Observatory (mini-swe-agent)

Local web app that runs mini-swe-agent and shows two panels side by side:

- **Left:** an Archify map of the agent loop, marking where the LLM call and the tool (bash) calls happen.
- **Right:** the run's call history, grouped per step. Each step shows the thought, the LLM call with the bash commands it chose, each command's execution (exit code, duration, submission), format errors, and the observation.

Like the browser-use and smolagents dashboards, the history comes from boundary logs that are placed directly in the agent loop (`src/minisweagent/agents/default.py`).

## Run

Same steps as the browser-use dashboard:

```bash
cp .env.example .env      # then set ANTHROPIC_API_KEY in .env
uv sync
uv run uvicorn examples.agent_dashboard.app:app --port 8000
```

Open http://127.0.0.1:8000. The key is read from the repository `.env` on the server and never sent to the browser.

- `uv sync` installs the default `dev` and `dashboard` dependency groups (FastAPI, uvicorn) from `pyproject.toml`. The upstream `.gitignore` ignores `uv.lock`, so the lock file stays local.
- In the model field you can enter a plain Claude name such as `claude-sonnet-5`, and the app adds the `anthropic/` prefix. Other providers take a LiteLLM name such as `openai/gpt-5`, plus that provider's key in `.env`.
- The agent uses the built-in `mini.yaml` config with `LocalEnvironment`. **The agent's bash commands run on this machine.** Each run gets a fresh working directory, `runs/<run_id>-workspace`, so its commands don't start inside the repository.
- Cost tracking is set to `ignore_errors`. For models that LiteLLM has no price for, the cost shows as $0, so the agent's cost limit doesn't apply. Use the step limit instead.

## How tracing works

The boundaries log at DEBUG level on the `agent` logger, which `DefaultAgent` already uses. That logger shows nothing by default, so the `mini` CLI output doesn't change. The tags follow the browser-use and smolagents forks:

| Log | Where | Args |
|---|---|---|
| `[LLM_QUERY]` | `DefaultAgent.query`, before `model.query` (after the step/cost/time limit checks) | `step, model, messages` |
| `[LLM_RESPONSE]` | after the model returns parsed actions | `step, actions, cost` |
| `[LLM_ERROR]` | when `model.query` raises, including `FormatError` for an unparsable reply | `step, error_type` |
| `[TOOL_EXECUTION_START/END/ERROR]` | `DefaultAgent.execute_action` around `env.execute` | `action, params` / `action, returncode, duration` / `action, error_type, duration` |

How the events are assembled:

- `execute_action` is the only place where actions run. `InteractiveAgent`, which the `mini` CLI uses, calls it too. Before this change, `InteractiveAgent` called `env.execute` directly.
- Submitting the task shows up as `[TOOL_EXECUTION_ERROR] error_type=Submitted`, because the environment signals completion by raising `Submitted`. The UI shows it as a successful submission.
- While a run is active, `DashboardLogHandler` in `app.py` is attached to the `agent` and `minisweagent` loggers. It builds trace events from each record's positional `args`. If you change the argument order of those log calls, update `parse_boundary_record` as well.
- mini-swe-agent has no step callback API. Step results come from `DashboardAgent`, a `DefaultAgent` subclass whose `add_messages` override reports the thought, observations, format errors and the exit message. Every message goes through `add_messages`.

## API

- `POST /api/run` starts a run. The body is `{task, max_steps, model}`. Only one run can be active at a time.
- `GET /api/events` is an SSE stream. It replays the current run's events and then streams new ones live.
- `GET /api/runs/current/trace.json` returns the full run record. Each finished run is also saved to `runs/<run_id>.json`, which is gitignored.

## Tests

```bash
uv run pytest tests/agents/test_boundary_logging.py examples/agent_dashboard/test_app.py
```

`tests/agents/test_boundary_logging.py` checks the core log lines for `DefaultAgent` and `InteractiveAgent`, including submission and format errors. `examples/agent_dashboard/test_app.py` runs `execute_agent` end to end and checks the trace events built from those logs. Both use `DeterministicModel` and the real `LocalEnvironment`, without mocks.

## Known gaps

- Only `LocalEnvironment` is wired into the dashboard. Other environments (Docker, Singularity, SWE-ReX) also go through `execute_action`, so they would be traced the same way, but `build_model`/`execute_agent` would need to choose them.
- The cost of a reply that fails to parse (`FormatError`) isn't in `[LLM_RESPONSE]`, because there is no response log for it. `DefaultAgent.run` still adds that cost to the run total.

## Architecture diagram

`static/agent-architecture.html` is a self-contained Archify (3.0.1) export generated from `agent-architecture.json`. Its source links point to `meta.repository.revision`, the fork commit whose line numbers match the `sources[].line` values. If you edit the logging in `agents/default.py`, update those line numbers and the revision together.

After editing the spec, validate it and then regenerate the HTML. Run these from the Archify repo's `archify/` directory:

```bash
node bin/archify.mjs validate architecture <mini-swe-agent>/examples/agent_dashboard/agent-architecture.json \
  --quality showcase --repo-root <mini-swe-agent>
node bin/archify.mjs deliver architecture <mini-swe-agent>/examples/agent_dashboard/agent-architecture.json \
  <mini-swe-agent>/examples/agent_dashboard/static/agent-architecture.html \
  --quality showcase --repo-root <mini-swe-agent> --json
```

`deliver` also writes an `agent-architecture.delivery.json` receipt that contains absolute local paths. Delete it rather than committing it. The `showcase` profile checks layout, and the `viewBox` width must stay at or below 1240 so the labels stay readable in the dashboard panel.

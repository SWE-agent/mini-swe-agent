## This step: run the agent once and read its log

The logging is in place. Check it at runtime:

1. Write a script `{kit}/runs/check_run.py` that runs the agent's run function (`{entry}`) on a short task with a scripted model: replace the model client with a small object that returns a reply asking for one tool call, then a final answer, so that no API key or network is needed. Answer any user prompt from the script.
2. Run it with `uv run python {kit}/runs/check_run.py`. The environment variable `OBSERVATORY_RUNS` is already set to `{kit}/runs`, so do not set it yourself.
3. Read the JSON-lines file it created in `{kit}/runs/` and confirm that the events come in order: `user_input` (`tool_start`/`tool_end`), `llm_query`, `llm_response`, `tool_start`, `tool_end`, then the next `llm_query`.

If an event is missing or the logging raises, fix the logger calls (only those). Leave the script and the log file in `{kit}/runs/`.

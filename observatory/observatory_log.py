"""Agent Observatory logger: one tiny, agent-agnostic way for an agent to report its LLM calls and tool executions.

Copy this file unchanged into the agent's main package and call the functions below where the agent calls the model and
where it runs a tool/action. Each call appends one JSON line to a run file that the dashboard (observatory/dashboard.py)
shows live. Nothing is written, and the agent behaves exactly as before, unless the environment variable
OBSERVATORY_RUNS is set to a directory (one file per process is created in it).

    llm_query(step, model=None, messages=None, **fields)      right before every model call
    llm_response(step, actions=None, **fields)                after the reply was parsed into actions (a list; [] if none)
    llm_error(step, error, **fields)                          when the model call raised
    start = tool_start(action, params=None, step=None)        right before one tool/action runs; keep the return value
    tool_end(start, action, result=None, step=None, **fields) after it returned; `result` is a short outcome
    tool_error(start, action, error, step=None, **fields)     when it raised

`step` is the agent's own step counter. A tool event without `step` belongs to the latest llm_query. Any extra keyword
field (cost, input_tokens, purpose, ...) is shown next to the event, so pass what helps someone reading the history.

Environment variables whose name contains KEY, TOKEN, SECRET or PASSWORD are scrubbed from every line, but secrets the
agent keeps elsewhere (for example in its own config) are not known here: never pass them as fields.
"""

import json
import os
import threading
import time
from pathlib import Path

MAX_CHARS = 4000
_lock = threading.Lock()
_files: dict[str, Path] = {}


def _secrets() -> list[str]:
    return [
        v
        for k, v in os.environ.items()
        if len(v) >= 6 and any(w in k.upper() for w in ("KEY", "TOKEN", "SECRET", "PASSWORD"))
    ]


def _clean(value, secrets: list[str]):
    if isinstance(value, str):
        for secret in secrets:
            value = value.replace(secret, "<redacted>")
        return value if len(value) <= MAX_CHARS else value[:MAX_CHARS] + "…"
    if isinstance(value, dict):
        return {str(k): _clean(v, secrets) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_clean(v, secrets) for v in value]
    if value is None or isinstance(value, (bool, int, float)):
        return value
    return _clean(str(value), secrets)


def _emit(kind: str, step: int | None, data: dict) -> None:
    runs = os.environ.get("OBSERVATORY_RUNS")
    if not runs:
        return
    with _lock:
        if runs not in _files:
            Path(runs).mkdir(parents=True, exist_ok=True)
            _files[runs] = Path(runs) / f"{time.strftime('%Y%m%d-%H%M%S')}-{os.getpid()}.jsonl"
        line = json.dumps(
            _clean({"t": time.time(), "kind": kind, "step": step, "data": data}, _secrets()), ensure_ascii=False
        )
        with _files[runs].open("a", encoding="utf-8") as file:
            file.write(line + "\n")


def llm_query(step: int, model: str | None = None, messages: int | None = None, **fields) -> None:
    _emit("llm_query", step, {"model": model, "messages": messages, **fields})


def llm_response(step: int, actions: list | None = None, **fields) -> None:
    _emit("llm_response", step, {"actions": actions or [], **fields})


def llm_error(step: int, error: BaseException, **fields) -> None:
    _emit("llm_error", step, {"error_type": type(error).__name__, "error": str(error), **fields})


def tool_start(action: str, params=None, step: int | None = None) -> float:
    _emit("tool_start", step, {"action": action, "params": params})
    return time.perf_counter()


def tool_end(start: float, action: str, result=None, step: int | None = None, **fields) -> None:
    _emit(
        "tool_end",
        step,
        {"action": action, "result": result, "duration_seconds": round(time.perf_counter() - start, 3), **fields},
    )


def tool_error(start: float, action: str, error: BaseException, step: int | None = None, **fields) -> None:
    _emit(
        "tool_error",
        step,
        {
            "action": action,
            "error_type": type(error).__name__,
            "error": str(error),
            "duration_seconds": round(time.perf_counter() - start, 3),
            **fields,
        },
    )

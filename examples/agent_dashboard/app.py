"""Local dashboard: run mini-swe-agent and stream its LLM/tool call history next to an architecture map.

The history comes from the boundary logs in the agent loop itself ([LLM_QUERY], [LLM_RESPONSE], [LLM_ERROR] and
[TOOL_EXECUTION_*] in agents/default.py), the same way the browser-use and smolagents dashboards read their logs.
"""

import asyncio
import json
import logging
import os
import threading
import time
import uuid
from collections.abc import AsyncIterator, Callable
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from pydantic import BaseModel, ConfigDict, Field

from minisweagent import Model
from minisweagent.agents.default import DefaultAgent
from minisweagent.config import get_config_from_spec
from minisweagent.environments import get_environment
from minisweagent.models import get_model

APP_DIR = Path(__file__).resolve().parent
# Both agent dashboards serve the same paths on the same port; make browsers revalidate instead of reusing the other one's page.
NO_CACHE = {"Cache-Control": "no-cache"}
load_dotenv(APP_DIR.parents[1] / ".env")

TraceKind = Literal[
    "run", "llm_query", "llm_response", "llm_error", "tool_start", "tool_end", "tool_error", "step", "log"
]
TAG_TO_KIND: dict[str, TraceKind] = {
    "[LLM_QUERY]": "llm_query",
    "[LLM_RESPONSE]": "llm_response",
    "[LLM_ERROR]": "llm_error",
    "[TOOL_EXECUTION_START]": "tool_start",
    "[TOOL_EXECUTION_END]": "tool_end",
    "[TOOL_EXECUTION_ERROR]": "tool_error",
}
PREVIEW_CHARS = 2000


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def preview(value: Any, limit: int = PREVIEW_CHARS) -> str | None:
    if value is None:
        return None
    text = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False, default=str)
    return text if len(text) <= limit else text[:limit] + "…"


class RunRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    task: str = Field(min_length=3, max_length=4000)
    max_steps: int = Field(default=15, ge=1, le=100)
    model: str = Field(default="claude-sonnet-5", min_length=1, max_length=100)


class TraceEvent(BaseModel):
    seq: int
    run_id: str
    timestamp: str
    kind: TraceKind
    level: str = "INFO"
    step: int | None = None
    message: str
    data: dict[str, Any] = Field(default_factory=dict)


class RunRecord(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    task: str
    model: str
    max_steps: int
    status: Literal["running", "finished", "failed"] = "running"
    started_at: str = Field(default_factory=_now)
    finished_at: str | None = None
    events: list[TraceEvent] = Field(default_factory=list)


class TraceHub:
    """Owns the current run, saves it under `runs_dir` and fans its events out. Safe to publish from any thread."""

    def __init__(self, runs_dir: Path) -> None:
        self.runs_dir = runs_dir
        self.run: RunRecord | None = None
        self.current_step: int | None = None
        self._llm_started_at: float | None = None
        self._subscribers: set[Callable[[TraceEvent], None]] = set()
        self._lock = threading.Lock()

    def start(self, request: RunRequest) -> RunRecord:
        with self._lock:
            self.run = RunRecord(task=request.task, model=request.model, max_steps=request.max_steps)
            self.current_step = None
            self._llm_started_at = None
            return self.run

    def workspace(self) -> Path:
        """Fresh working directory for the current run, so the agent's shell commands don't touch the repo."""
        assert self.run is not None
        path = self.runs_dir / f"{self.run.id}-workspace"
        path.mkdir(parents=True, exist_ok=True)
        return path

    def subscribe(self, callback: Callable[[TraceEvent], None]) -> list[TraceEvent]:
        """Register a subscriber and return the events published so far, atomically."""
        with self._lock:
            self._subscribers.add(callback)
            return list(self.run.events) if self.run else []

    def unsubscribe(self, callback: Callable[[TraceEvent], None]) -> None:
        with self._lock:
            self._subscribers.discard(callback)

    def publish(
        self,
        kind: TraceKind,
        message: str,
        *,
        level: str = "INFO",
        step: int | None = None,
        data: dict[str, Any] | None = None,
    ) -> TraceEvent:
        with self._lock:
            assert self.run is not None, "publish() called without an active run"
            data = dict(data or {})
            if kind == "llm_query":
                self.current_step = step
                self._llm_started_at = time.monotonic()
            elif kind in ("llm_response", "llm_error") and self._llm_started_at is not None:
                data["duration_seconds"] = round(time.monotonic() - self._llm_started_at, 3)
                self._llm_started_at = None
            if step is None and kind in ("tool_start", "tool_end", "tool_error"):
                step = self.current_step
            event = TraceEvent(
                seq=len(self.run.events),
                run_id=self.run.id,
                timestamp=_now(),
                kind=kind,
                level=level,
                step=step,
                message=message,
                data=data,
            )
            self.run.events.append(event)
            subscribers = tuple(self._subscribers)
        for callback in subscribers:
            callback(event)
        return event

    def finish(self, status: Literal["finished", "failed"]) -> None:
        with self._lock:
            assert self.run is not None
            self.run.status = status
            self.run.finished_at = _now()
            self.runs_dir.mkdir(parents=True, exist_ok=True)
            (self.runs_dir / f"{self.run.id}.json").write_text(self.run.model_dump_json(indent=2), encoding="utf-8")


def parse_boundary_record(kind: TraceKind, args: tuple[Any, ...]) -> tuple[int | None, str, dict[str, Any]]:
    """Turn the positional args of a boundary log call in agents/default.py into structured trace data."""
    if kind == "llm_query":
        step, model, n_messages = args
        return int(step), f"{model} · {n_messages} messages", {"model": model, "messages": n_messages}
    if kind == "llm_response":
        step, actions, cost = args
        commands = [action.get("command", "") for action in actions]
        return int(step), f"bash × {len(commands)}" if commands else "no action", {"commands": commands, "cost": cost}
    if kind == "llm_error":
        step, error_type = args
        return int(step), f"model call failed: {error_type}", {"error_type": error_type}
    if kind == "tool_start":
        action, params = args
        return None, action, {"action": action, "command": params.get("command", "")}
    action, result, duration = args
    key = "returncode" if kind == "tool_end" else "error_type"
    return None, action, {"action": action, key: result, "duration_seconds": round(float(duration), 3)}


class DashboardLogHandler(logging.Handler):
    """Convert mini-swe-agent log records into trace events."""

    def __init__(self, hub: TraceHub) -> None:
        super().__init__(logging.DEBUG)
        self.hub = hub

    def emit(self, record: logging.LogRecord) -> None:
        try:
            boundary: TraceKind | None = next(
                (k for tag, k in TAG_TO_KIND.items() if isinstance(record.msg, str) and record.msg.startswith(tag)),
                None,
            )
            kind: TraceKind
            step: int | None = None
            if boundary is not None:
                kind = boundary
                step, message, data = parse_boundary_record(kind, record.args if isinstance(record.args, tuple) else ())
            elif record.levelno >= logging.WARNING:
                kind = "log"
                message, data = record.getMessage(), {"logger": record.name}
            else:
                return
        except Exception:
            self.handleError(record)
            return
        self.hub.publish(kind, message, level=record.levelname, step=step, data=data)


class DashboardAgent(DefaultAgent):
    """DefaultAgent that also reports what each step produced: the thought, observations, format errors and exit."""

    def __init__(self, *args, hub: TraceHub, **kwargs):
        super().__init__(*args, **kwargs)
        self.hub = hub

    def add_messages(self, *messages: dict) -> list[dict]:
        for message in messages:
            if self.n_calls > 0:
                self._publish_step_message(message)
        return super().add_messages(*messages)

    def _publish_step_message(self, message: dict) -> None:
        extra = message.get("extra", {})
        if message.get("role") == "exit":
            data = {
                "type": "exit",
                "exit_status": extra.get("exit_status"),
                "submission": preview(extra.get("submission")),
            }
        elif message.get("role") == "assistant":
            data = {"type": "thought", "content": preview(message.get("content"))}
        elif extra.get("interrupt_type"):
            data = {
                "type": "interrupt",
                "interrupt_type": extra["interrupt_type"],
                "content": preview(message.get("content")),
            }
        else:
            data = {"type": "observation", "content": preview(message.get("content"))}
        self.hub.publish("step", data["type"], step=self.n_calls, data=data)


def litellm_model_name(model: str) -> str:
    """Accept plain Claude names (as in the browser-use dashboard); other models need a LiteLLM `provider/model` name."""
    return f"anthropic/{model}" if "/" not in model and model.startswith("claude") else model


def build_model(request: RunRequest, config: dict) -> Model:
    return get_model(litellm_model_name(request.model), config["model"] | {"cost_tracking": "ignore_errors"})


def missing_api_key(model: str) -> str | None:
    if litellm_model_name(model).startswith("anthropic/") and not os.getenv("ANTHROPIC_API_KEY"):
        return "ANTHROPIC_API_KEY"
    return None


def execute_agent(hub: TraceHub, request: RunRequest, model: Model | None = None) -> None:
    """Blocking agent run with the boundary logs routed into the hub; executed in a worker thread."""
    config = get_config_from_spec("mini.yaml")
    loggers = [logging.getLogger("agent"), logging.getLogger("minisweagent")]
    previous_levels = [logger.level for logger in loggers]
    handler = DashboardLogHandler(hub)
    for logger in loggers:
        logger.addHandler(handler)
        logger.setLevel(logging.DEBUG)
    status: Literal["finished", "failed"] = "failed"
    try:
        agent = DashboardAgent(
            model or build_model(request, config),
            get_environment(config["environment"] | {"cwd": str(hub.workspace())}, default_type="local"),
            hub=hub,
            **{k: v for k, v in config["agent"].items() if k != "mode"} | {"step_limit": request.max_steps},
        )
        info = agent.run(request.task)
        status = "finished"
        hub.publish(
            "run",
            f"Run finished: exit_status={info.get('exit_status')} steps={agent.n_calls}",
            data={
                "exit_status": info.get("exit_status"),
                "success": True if info.get("exit_status") == "Submitted" else None,
                "final_result": preview(info.get("submission"), 4000),
                "cost": agent.cost,
            },
        )
    except Exception as error:
        hub.publish("run", f"Run failed: {type(error).__name__}: {preview(str(error), 600)}", level="ERROR")
    finally:
        for logger, level in zip(loggers, previous_levels):
            logger.removeHandler(handler)
            logger.setLevel(level)
        hub.finish(status)


hub = TraceHub(APP_DIR / "runs")
run_task: asyncio.Task[None] | None = None


async def run_in_background(request: RunRequest) -> None:
    global run_task
    try:
        await asyncio.to_thread(execute_agent, hub, request)
    finally:
        run_task = None


app = FastAPI(title="mini-swe-agent Agent Observatory")


@app.get("/")
async def index() -> FileResponse:
    return FileResponse(APP_DIR / "static" / "index.html", headers=NO_CACHE)


@app.get("/architecture")
async def architecture() -> FileResponse:
    return FileResponse(APP_DIR / "static" / "agent-architecture.html", headers=NO_CACHE)


@app.get("/api/status")
async def status() -> dict[str, Any]:
    return {
        "missing_api_key": missing_api_key(RunRequest.model_fields["model"].default),
        "running": bool(run_task and not run_task.done()),
        "run": hub.run.model_dump(exclude={"events"}) if hub.run else None,
    }


@app.post("/api/run", status_code=202)
async def start_run(request: RunRequest) -> dict[str, str]:
    global run_task
    if key := missing_api_key(request.model):
        raise HTTPException(status_code=400, detail=f"{key} is missing from .env")
    if run_task and not run_task.done():
        raise HTTPException(status_code=409, detail="An agent is already running")
    run = hub.start(request)
    hub.publish("run", f"Run started: {request.task}", data=request.model_dump())
    run_task = asyncio.create_task(run_in_background(request))
    return {"status": "started", "run_id": run.id}


@app.get("/api/runs/current/trace.json")
async def current_trace() -> JSONResponse:
    if hub.run is None:
        raise HTTPException(status_code=404, detail="No run yet")
    return JSONResponse(
        hub.run.model_dump(mode="json"),
        headers={"Content-Disposition": f'attachment; filename="mini-swe-agent-trace-{hub.run.id}.json"'},
    )


@app.get("/api/events")
async def events() -> StreamingResponse:
    """SSE: replay the current run's history, then stream new events live."""
    loop = asyncio.get_running_loop()
    queue: asyncio.Queue[TraceEvent] = asyncio.Queue(maxsize=5000)

    def enqueue(event: TraceEvent) -> None:
        loop.call_soon_threadsafe(queue.put_nowait, event)

    async def stream() -> AsyncIterator[str]:
        backlog = hub.subscribe(enqueue)
        try:
            snapshot = hub.run.model_dump(mode="json", exclude={"events"}) if hub.run else None
            yield f"event: snapshot\ndata: {json.dumps(snapshot, ensure_ascii=False)}\n\n"
            for event in backlog:
                yield f"data: {event.model_dump_json()}\n\n"
            while True:
                try:
                    event = await asyncio.wait_for(queue.get(), timeout=15)
                    yield f"data: {event.model_dump_json()}\n\n"
                except TimeoutError:
                    yield ": keepalive\n\n"
        finally:
            hub.unsubscribe(enqueue)

    return StreamingResponse(stream(), media_type="text/event-stream", headers={"Cache-Control": "no-cache"})

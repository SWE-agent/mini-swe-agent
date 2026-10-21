"""Persistence for controlled, local resumes."""

import json
import math
import os
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Annotated, Literal
from uuid import UUID

from pydantic import ConfigDict, Field
from pydantic.dataclasses import dataclass

from minisweagent import __version__

PAUSE_STATUSES = {"Paused", "LimitsExceeded", "TimeExceeded"}


@dataclass(config=ConfigDict(extra="forbid"))
class ResumeState:
    version: Literal[1]
    status: Literal["running", "paused", "finished"]
    checkpoint_id: str
    parent_checkpoint_id: str | None
    run_dir: str
    template_vars: dict
    elapsed_seconds: Annotated[float, Field(ge=0, allow_inf_nan=False)]
    consecutive_format_errors: Annotated[int, Field(ge=0, strict=True)]


def atomic_write(path: Path, content: str) -> None:
    # The temporary file must share the destination's filesystem for atomic replacement.
    with NamedTemporaryFile(dir=path.parent, delete=False) as temporary:
        temporary_path = Path(temporary.name)
    try:
        temporary_path.write_text(content, encoding="utf-8")
        with temporary_path.open("rb") as stream:
            os.fsync(stream.fileno())
        temporary_path.replace(path)
    finally:
        temporary_path.unlink(missing_ok=True)


def validate_history(data: dict) -> None:
    messages = data["messages"]
    if [m.get("role") for m in messages[:2]] != ["system", "user"]:
        raise ValueError("Checkpoint is missing its initial messages")
    if messages[-1].get("role") != "exit" or messages[-1]["extra"]["exit_status"] not in PAUSE_STATUSES:
        raise ValueError("Checkpoint did not stop at a controlled pause")
    pending: list[str | None] = []
    calls, cost, errors = 0, 0.0, 0
    for message in messages[2:-1]:
        extra = message.get("extra", {})
        if pending:
            call_id = pending.pop(0)
            if (
                message.get("role") != ("tool" if call_id else "user")
                or message.get("tool_call_id") != call_id
                or "returncode" not in extra
                or extra["returncode"] < 0
                or extra.get("exception_info")
            ):
                raise ValueError("Checkpoint has missing or interrupted action results")
        elif message.get("role") == "assistant":
            pending = [action.get("tool_call_id") for action in extra["actions"]]
            tool_ids = [call["id"] for call in message.get("tool_calls") or []]
            if not pending or (tool_ids and (tool_ids != pending or len(set(tool_ids)) != len(tool_ids))):
                raise ValueError("Checkpoint has inconsistent actions and tool calls")
            if any(pending) and not tool_ids:
                raise ValueError("Checkpoint is missing tool calls")
            calls += 1
            cost += extra.get("cost", 0.0)
            errors = 0
        elif message.get("role") == "user" and extra.get("interrupt_type") == "FormatError":
            calls += 1
            cost += extra.get("cost", 0.0)
            errors += 1
        else:
            raise ValueError("Checkpoint contains an unexpected message")
    stats = data["info"]["model_stats"]
    if (
        pending
        or type(stats["api_calls"]) is not int
        or calls != stats["api_calls"]
        or not math.isfinite(stats["instance_cost"])
        or stats["instance_cost"] < 0
        or not math.isclose(cost, stats["instance_cost"])
        or errors != data["info"]["resume"]["consecutive_format_errors"]
    ):
        raise ValueError("Checkpoint has incomplete actions or inconsistent counters")


def load_checkpoint(run_dir: Path) -> tuple[dict, ResumeState]:
    run_dir = run_dir.resolve()
    checkpoint_id = UUID((run_dir / "latest").read_text(encoding="utf-8")).hex
    data = json.loads((run_dir / f"{checkpoint_id}.traj.json").read_text(encoding="utf-8"))
    state = ResumeState(**data["info"]["resume"])
    if (
        data["trajectory_format"] != "mini-swe-agent-1.1"
        or data["info"]["mini_version"] != __version__
        or state.status != "paused"
        or state.checkpoint_id != checkpoint_id
        or state.run_dir != str(run_dir)
    ):
        raise ValueError("Only a compatible, paused checkpoint in its original run directory can be resumed")
    validate_history(data)
    return data, state

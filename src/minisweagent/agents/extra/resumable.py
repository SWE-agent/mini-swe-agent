"""Resume a controlled stop without replaying actions or resetting task budgets."""

import json
import math
import time
from dataclasses import asdict
from pathlib import Path
from uuid import uuid4

from minisweagent import Model
from minisweagent.agents.default import DefaultAgent
from minisweagent.agents.utils.checkpoint import (
    PAUSE_STATUSES,
    ResumeState,
    atomic_write,
    load_checkpoint,
    validate_history,
)
from minisweagent.environments.local import LocalEnvironment
from minisweagent.exceptions import InterruptAgentFlow

BUDGET_FIELDS = {"step_limit", "cost_limit", "wall_time_limit_seconds"}


class ResumableAgent(DefaultAgent):
    def __init__(self, model: Model, env: LocalEnvironment, *, run_dir: Path, pause_after: int = 0, **kwargs):
        super().__init__(model, env, **kwargs)
        if type(env) is not LocalEnvironment:
            raise ValueError("Controlled resume requires LocalEnvironment")
        if pause_after < 0 or any(
            not math.isfinite(value := getattr(self.config, field)) or value < 0 for field in BUDGET_FIELDS
        ):
            raise ValueError("Budgets and pause_after must be finite and nonnegative")
        if self.config.output_path is not None:
            raise ValueError("Use run_dir instead of output_path for resumable runs")
        self.env.config.cwd = str(Path(env.config.cwd or Path.cwd()).resolve())
        if not Path(self.env.config.cwd).is_dir():
            raise ValueError("Working directory does not exist")
        self.run_dir = run_dir.resolve()
        self.pause_after = pause_after
        self.checkpoint_id = uuid4().hex
        self.parent_checkpoint_id: str | None = None
        self.config.output_path = self.run_dir / f"{self.checkpoint_id}.traj.json"
        self._session_start_calls = self._calls_before_step = 0
        self._stop_time: float | None = None

    def run(self, task: str = "", **kwargs) -> dict:
        self.run_dir.mkdir(parents=True)
        self._start_time = time.time()
        return super().run(task, **kwargs)

    def resume(self) -> dict:
        data, state = load_checkpoint(self.run_dir)
        saved_config = data["info"]["config"]
        current_config = super().serialize()["info"]["config"]
        for config in (saved_config, current_config):
            config["agent"] = {
                key: value for key, value in config["agent"].items() if key not in BUDGET_FIELDS | {"output_path"}
            }
        if current_config != saved_config:
            raise ValueError("Agent, model, and environment configuration must match the checkpoint")
        stats = data["info"]["model_stats"]
        for field, used in (
            ("step_limit", stats["api_calls"]),
            ("cost_limit", stats["instance_cost"]),
            ("wall_time_limit_seconds", state.elapsed_seconds),
        ):
            if 0 < getattr(self.config, field) <= used:
                raise ValueError(f"{field} is exhausted; increase the cumulative limit before resuming")
        # A claim survives crashes: an old pause must never become reusable after new side effects.
        (self.run_dir / f"{state.checkpoint_id}.claimed").mkdir()
        self.checkpoint_id = uuid4().hex
        self.config.output_path = self.run_dir / f"{self.checkpoint_id}.traj.json"
        self.messages = data["messages"][:-1]
        self.cost = stats["instance_cost"]
        self.n_calls = self._session_start_calls = stats["api_calls"]
        self.n_consecutive_format_errors = state.consecutive_format_errors
        self.extra_template_vars = state.template_vars
        self._start_time = time.time() - state.elapsed_seconds
        self.parent_checkpoint_id = state.checkpoint_id
        return self._run_loop()

    def _run_loop(self) -> dict:
        self._stop_time = None
        self.save(self.config.output_path)
        atomic_write(self.run_dir / "latest", self.checkpoint_id)
        return super()._run_loop()

    def step(self) -> list[dict]:
        self._calls_before_step = self.n_calls
        if 0 < self.pause_after <= self.n_calls - self._session_start_calls:
            raise InterruptAgentFlow(
                {"role": "exit", "content": "Paused", "extra": {"exit_status": "Paused", "submission": ""}}
            )
        return super().step()

    def execute_actions(self, message: dict) -> list[dict]:
        outputs = []
        interrupted = False
        for action in message.get("extra", {}).get("actions", []):
            outputs.append(self.env.execute(action))
            # Stop the batch before later commands can act on an interrupted result.
            interrupted = bool(outputs[-1].get("exception_info") or outputs[-1]["returncode"] < 0)
            if interrupted:
                break
        observations = self.add_messages(
            *self.model.format_observation_messages(message, outputs, self.get_template_vars())
        )
        if interrupted:
            raise InterruptAgentFlow(
                {
                    "role": "exit",
                    "content": "Environment execution was interrupted",
                    "extra": {"exit_status": "EnvironmentError", "submission": ""},
                }
            )
        return observations

    def serialize(self, *extra_dicts) -> dict:
        exit_status = self.messages[-1].get("extra", {}).get("exit_status", "") if self.messages else ""
        status = "finished" if exit_status else "running"
        # Limits are resumable only if raised before query() increments the call counter.
        if exit_status in PAUSE_STATUSES and self.n_calls == self._calls_before_step:
            status = "paused"
        return super().serialize(
            *extra_dicts,
            {
                "info": {
                    "resume": asdict(
                        ResumeState(
                            version=1,
                            status=status,
                            checkpoint_id=self.checkpoint_id,
                            parent_checkpoint_id=self.parent_checkpoint_id,
                            run_dir=str(self.run_dir),
                            template_vars=self.extra_template_vars,
                            elapsed_seconds=max(
                                0.0,
                                (self._stop_time if self._stop_time is not None else time.time()) - self._start_time,
                            ),
                            consecutive_format_errors=self.n_consecutive_format_errors,
                        )
                    )
                }
            },
        )

    def save(self, path: Path | None, *extra_dicts) -> dict:
        if self._stop_time is None and self.messages and self.messages[-1].get("role") == "exit":
            self._stop_time = time.time()
        data = self.serialize(*extra_dicts)
        if data["info"]["resume"]["status"] == "paused":
            validate_history(data)
        if path:
            atomic_write(path, json.dumps(data, indent=2, allow_nan=False))
        return data

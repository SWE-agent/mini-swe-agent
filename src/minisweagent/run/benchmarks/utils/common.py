"""Shared agent utilities for benchmark runners."""

import time

from minisweagent.agents.default import DefaultAgent
from minisweagent.exceptions import LimitsExceeded
from minisweagent.run.benchmarks.utils.batch_progress import RunBatchProgressManager


class TimeExceeded(LimitsExceeded):
    """Raised when the agent has exceeded its wall-clock time limit."""


class TimeAwareMixin:
    """Mixin that adds wall-clock time tracking to any agent class."""

    def __init__(self, *args, wall_time_limit_seconds: int = 0, **kwargs):
        super().__init__(*args, **kwargs)
        self.wall_time_limit_seconds = wall_time_limit_seconds
        self._start_time = time.time()

    @property
    def elapsed_seconds(self) -> int:
        return int(time.time() - self._start_time)

    def get_template_vars(self, **kwargs) -> dict:
        return super().get_template_vars(
            elapsed_seconds=self.elapsed_seconds,
            wall_time_limit_seconds=self.wall_time_limit_seconds,
            **kwargs,
        )

    def step(self) -> list[dict]:
        if self.wall_time_limit_seconds > 0 and self.elapsed_seconds > self.wall_time_limit_seconds:
            raise TimeExceeded(
                {
                    "role": "exit",
                    "content": "TimeExceeded",
                    "extra": {"exit_status": "TimeExceeded", "submission": ""},
                }
            )
        return super().step()


class ProgressTrackingAgent(TimeAwareMixin, DefaultAgent):
    """Agent with progress tracking and optional wall-clock time limit."""

    def __init__(self, *args, progress_manager: RunBatchProgressManager, instance_id: str = "", **kwargs):
        super().__init__(*args, **kwargs)
        self.progress_manager = progress_manager
        self.instance_id = instance_id

    def step(self) -> dict:
        self.progress_manager.update_instance_status(self.instance_id, f"Step {self.n_calls + 1:3d} (${self.cost:.2f})")
        return super().step()

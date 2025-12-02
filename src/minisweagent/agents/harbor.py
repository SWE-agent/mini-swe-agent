"""Agent with incremental trajectory saving for Harbor/terminal-bench."""

from pathlib import Path

from minisweagent.agents.default import DefaultAgent
from minisweagent.run.utils.save import save_traj


class HarborMiniAgent(DefaultAgent):
    """Agent that saves trajectory after each turn.

    See: https://github.com/SWE-agent/mini-swe-agent/pull/585
    """

    def __init__(self, *args, traj_path: Path | None = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.traj_path = traj_path

    def step(self):
        """Execute one step and save trajectory."""
        try:
            result = super().step()
        finally:
            if self.traj_path:
                save_traj(self, self.traj_path, print_path=False)
        return result

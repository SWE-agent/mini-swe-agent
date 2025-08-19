import logging
import os
import platform
import shutil
import subprocess
import tempfile
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class BubblewrapEnvironmentConfig:
    cwd: str = ""
    env: dict[str, str] = field(default_factory=dict)
    timeout: int = 30
    executable: str = os.getenv("MSWEA_BUBBLEWRAP_EXECUTABLE", "bwrap")
    """Path to the bubblewrap executable."""
    wrapper_args: list[str] = []
    """Arguments to pass to the bubblewrap executable."""


class BubblewrapEnvironment:
    def __init__(self, *, config_class: type = BubblewrapEnvironmentConfig, **kwargs):
        """This class executes bash commands in a bubblewrap environment and a separate working
        directory for each environment. See `BubblewrapEnvironmentConfig` for kwargs.
        """
        self.logger = logging.getLogger("minisweagent.environment")
        self.config = config_class(**kwargs)
        self.working_dir = Path(tempfile.gettempdir()) / f"minisweagent-{uuid.uuid4().hex[:8]}"

    def execute(self, command: str, cwd: str = ""):
        """Execute a command in the bubblewrap environment and return the result as a dict."""
        cwd = cwd or self.config.cwd or self.working_dir

        cmd = [self.config.executable] + self.config.wrapper_args + ["bash", "-c", command]
        result = subprocess.run(
            cmd,
            text=True,
            cwd=cwd,
            timeout=self.config.timeout,
            encoding="utf-8",
            errors="replace",
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
        return {"output": result.stdout, "returncode": result.returncode}

    def cleanup(self):
        if self.working_dir.exists():
            shutil.rmtree(self.working_dir)

    def __del__(self):
        """Cleanup working_dir when object is destroyed."""
        self.cleanup()

    def get_template_vars(self) -> dict[str, Any]:
        return asdict(self.config) | platform.uname()._asdict()

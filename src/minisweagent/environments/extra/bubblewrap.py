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
    wrapper_args: list[str] = field(
        default_factory=lambda: [
            "--ro-bind",
            "/usr",
            "/usr",
            "--ro-bind",
            "/bin",
            "/bin",
            "--ro-bind",
            "/lib",
            "/lib",
            "--tmpfs",
            "/tmp",
            "--proc",
            "/proc",
            "--dev",
            "/dev",
            "--unshare-all",
            "--share-net",
        ]
    )
    """Arguments to pass to the bubblewrap executable."""


class BubblewrapEnvironment:
    def __init__(self, *, config_class: type = BubblewrapEnvironmentConfig, **kwargs):
        """This class executes bash commands in a bubblewrap environment and a separate working
        directory for each environment. See `BubblewrapEnvironmentConfig` for kwargs.
        """
        self.logger = logging.getLogger("minisweagent.environment")
        self.config = config_class(**kwargs)
        self.working_dir = Path(tempfile.gettempdir()) / f"minisweagent-{uuid.uuid4().hex[:8]}"
        self.working_dir.mkdir(parents=True, exist_ok=True)

    def execute(self, command: str, cwd: str = "") -> dict[str, Any]:
        """Execute a command in the bubblewrap environment and return the result as a dict."""
        cwd = cwd or self.config.cwd or str(self.working_dir)

        cmd = [self.config.executable] + self.config.wrapper_args + ["--bind", cwd, cwd, "--chdir", cwd]

        # Add environment variables
        for key, value in self.config.env.items():
            cmd.extend(["--setenv", key, value])

        cmd.extend(["bash", "-c", command])

        result = subprocess.run(
            cmd,
            text=True,
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

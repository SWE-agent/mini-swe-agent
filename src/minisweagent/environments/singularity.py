#!/usr/bin/env python3

import os
import subprocess
from dataclasses import dataclass, field
import tempfile
from typing import Any
import uuid


@dataclass
class SingularityEnvironmentConfig:
    image: str
    cwd: str = "/"
    env: dict[str, str] = field(default_factory=dict)
    """Environment variables to set in the container."""
    forward_env: list[str] = field(default_factory=list)
    """Environment variables to forward to the container."""
    timeout: int = 30
    """Timeout for executing commands in the container."""
    executable: str = "singularity"
    """Path to the singularity executable."""


class SingularityEnvironment:
    def __init__(self, **kwargs):
        """Singularity environment. See `SingularityEnvironmentConfig` for kwargs."""
        self.config = SingularityEnvironmentConfig(**kwargs)
        self.sandbox_dir = os.path.join(tempfile.gettempdir(), f"minisweagent-{uuid.uuid4().hex[:8]}")

        subprocess.run(
            [self.config.executable, "build", "--sandbox", self.sandbox_dir, self.config.image],
            check=True,
        )

    def execute(self, command: str, cwd: str = "") -> dict[str, Any]:
        """Execute a command in a Singularity container and return the result as a dict."""
        cmd = [self.config.executable, "exec"]

        # Do not inherit directories and env vars from host
        cmd.extend(["--contain", "--cleanenv"])

        work_dir = cwd or self.config.cwd
        if work_dir and work_dir != "/":
            cmd.extend(["--pwd", work_dir])

        for key in self.config.forward_env:
            if (value := os.getenv(key)) is not None:
                cmd.extend(["--env", f"{key}={value}"])
        for key, value in self.config.env.items():
            cmd.extend(["--env", f"{key}={value}"])

        cmd.extend(["--writable", self.sandbox_dir, "bash", "-c", command])
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

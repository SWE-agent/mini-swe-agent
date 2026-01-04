#!/usr/bin/env python3

import logging
import os
import shutil
import subprocess
import tempfile
import time
import uuid
from pathlib import Path
from typing import Any

from jinja2 import StrictUndefined, Template
from pydantic import BaseModel

from minisweagent.exceptions import ExecutionTimeoutError, Submitted
from minisweagent.utils.serialize import recursive_merge


class SingularityEnvironmentConfig(BaseModel):
    image: str
    cwd: str = "/"
    env: dict[str, str] = {}
    """Environment variables to set in the container."""
    forward_env: list[str] = []
    """Environment variables to forward to the container."""
    timeout: int = 30
    """Timeout for executing commands in the container."""
    executable: str = os.getenv("MSWEA_SINGULARITY_EXECUTABLE", "singularity")
    """Path to the singularity executable."""
    sandbox_build_retries: int = 3
    """Number of retries for building the sandbox if an error occurs."""
    action_observation_template: str = (
        "<returncode>{{output.returncode}}</returncode>\n<output>\n{{output.output}}</output>"
    )
    """Template used to render the observation after executing an action."""
    timeout_template: str = "Command timed out. Output:\n{{output}}"
    """Template used when a command timed out."""


class SingularityEnvironment:
    def __init__(
        self, *, config_class: type = SingularityEnvironmentConfig, logger: logging.Logger | None = None, **kwargs
    ):
        """Singularity environment. See `SingularityEnvironmentConfig` for kwargs."""
        self.logger = logger or logging.getLogger("minisweagent.environment")
        self.config = config_class(**kwargs)
        self.sandbox_dir = self._build_sandbox()

    def _build_sandbox(self) -> Path:
        # Building the sandbox can fail (very rarely), so we retry it
        max_retries = self.config.sandbox_build_retries
        for attempt in range(max_retries):
            sandbox_dir = Path(tempfile.gettempdir()) / f"minisweagent-{uuid.uuid4().hex[:8]}"
            try:
                subprocess.run(
                    [self.config.executable, "build", "--sandbox", sandbox_dir, self.config.image],
                    check=True,
                    capture_output=True,
                )
                break
            except subprocess.CalledProcessError as e:
                shutil.rmtree(sandbox_dir, ignore_errors=True)
                self.logger.error(
                    f"Error building image {self.config.image}, stdout: {e.stdout}, stderr: {e.stderr} (attempt {attempt + 1}/{max_retries})"
                )
                if attempt == max_retries - 1:
                    raise
        return sandbox_dir

    def get_template_vars(self, **kwargs) -> dict[str, Any]:
        return recursive_merge(self.config.model_dump(), kwargs)

    def serialize(self) -> dict:
        return {
            "info": {
                "config": {
                    "environment": self.config.model_dump(mode="json"),
                    "environment_type": f"{self.__class__.__module__}.{self.__class__.__name__}",
                }
            }
        }

    def execute(self, command: str, cwd: str = "", *, timeout: int | None = None) -> dict[str, Any]:
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

        cmd.extend(["--writable", str(self.sandbox_dir), "bash", "-c", command])
        result = subprocess.run(
            cmd,
            text=True,
            timeout=timeout or self.config.timeout,
            encoding="utf-8",
            errors="replace",
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
        return {"output": result.stdout, "returncode": result.returncode}

    def execute_messages(self, messages: list[dict], extra_template_vars: dict[str, Any] | None = None) -> list[dict]:
        """Execute all actions in messages and return observation messages."""
        results = []
        for msg in messages:
            if "action" not in msg.get("extra", {}):
                continue
            action = msg["extra"]["action"]
            try:
                output = self.execute(action)
            except (TimeoutError, subprocess.TimeoutExpired) as e:
                output_text = e.output.decode("utf-8", errors="replace") if getattr(e, "output", None) else ""
                raise ExecutionTimeoutError(
                    Template(self.config.timeout_template, undefined=StrictUndefined).render(
                        **self.get_template_vars(action=action, output=output_text, **(extra_template_vars or {}))
                    )
                )
            self._check_finished(output)
            results.extend(self._get_observation_message(msg, output))
        return results

    def _get_observation_message(self, msg: dict, output: dict) -> list[dict]:
        """Get observation message for the output of an action."""
        content = Template(self.config.action_observation_template, undefined=StrictUndefined).render(
            **self.get_template_vars(action=msg["extra"]["action"], output=output)
        )
        return [
            {
                "role": "user",
                "content": content,
                "extra": {"raw_output": output["output"], "returncode": output["returncode"], "timestamp": time.time()},
            }
        ]

    def _check_finished(self, output: dict):
        """Raises Submitted exception if the output indicates task completion."""
        lines = output.get("output", "").rstrip().splitlines(keepends=True)
        if lines and lines[-1].strip() == "COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT":
            raise Submitted("".join(lines[:-1]))

    def cleanup(self):
        shutil.rmtree(self.sandbox_dir, ignore_errors=True)

    def __del__(self):
        """Cleanup sandbox when object is destroyed."""
        self.cleanup()

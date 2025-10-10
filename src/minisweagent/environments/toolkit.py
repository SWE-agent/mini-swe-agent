import logging
import os
import shlex
import subprocess
import uuid
from dataclasses import asdict, dataclass, field
from typing import Any
import time
import sys

@dataclass
class ToolkitEnvironmentConfig:
    image: str
    cwd: str = "/"
    """Working directory in which to execute commands."""
    env: dict[str, str] = field(default_factory=dict)
    """Environment variables to set in the container."""
    forward_env: list[str] = field(default_factory=list)
    """Environment variables to forward to the container.
    Variables are only forwarded if they are set in the host environment.
    In case of conflict with `env`, the `env` variables take precedence.
    """
    timeout: int = 30
    """Timeout for executing commands in the container."""
    executable: str = os.getenv("MSWEA_DOCKER_EXECUTABLE", "docker")
    """Path to the docker/container executable."""
    run_args: list[str] = field(default_factory=lambda: ["--rm"])
    """Additional arguments to pass to the docker/container executable.
    Default is ["--rm"], which removes the container after it exits.
    """
    container_timeout: str = "2h"
    """Max duration to keep container running. Uses the same format as the sleep command."""
    pull_timeout: int = 120
    """Timeout in seconds for pulling images."""
    account_name: str = "snow.core_llm.training04"

class ToolkitEnvironment:
    def __init__(self, *, config_class: type = ToolkitEnvironmentConfig, logger: logging.Logger | None = None, **kwargs):
        """This class executes bash commands in a Docker container using direct docker commands.
        See `ToolkitEnvironmentConfig` for keyword arguments.
        """
        self.logger = logger or logging.getLogger("minisweagent.environment")
        self.job_id: str | None = None
        self.config = config_class(**kwargs)
        self._start_toolkit_job()

    def get_template_vars(self) -> dict[str, Any]:
        return asdict(self.config)

    def _run_command(self, command):
        """Run a shell command and return the output as a string."""
        result = subprocess.run(command, shell=True, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"Error running command: {command}\n{result.stderr}", file=sys.stderr)
            sys.exit(1)
        return result.stdout.strip()

    def _start_toolkit_job(self):
        """Start the Docker container and return the container ID."""
        job_name = f"minisweagent-{uuid.uuid4().hex[:8]}".replace("-", "_")
        cmd = [
            "eai",
            "job",
            "new",
            "--image",
            self.config.image,
            "--name",
            job_name,
            "--account",
            self.config.account_name,
            "--",
            "/bin/bash",
            "-c",
            "while true; do sleep 1; done"
        ]
        # print(f"Starting toolkit job with command: {shlex.join(cmd)}")
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=self.config.pull_timeout,  # docker pull might take a while
            check=True,
        )
        self.job_id = self._run_command(f"eai job ls --account {self.config.account_name}| grep {job_name} | awk '{{print $1}}'")
        while self._run_command(f"eai job get {self.job_id} --field state") != "RUNNING":
            time.sleep(3)
        self._run_command(f"eai job exec {self.job_id} -- bash -c 'git config --global --add safe.directory /testbed && git status'")

        # print(f"Started toolkit for {job_name}. Job id: {self.job_id}")

    def execute(self, command: str, cwd: str = "", *, timeout: int | None = None) -> dict[str, Any]:
        """Execute a command in the Docker container and return the result as a dict."""
        cwd = cwd or self.config.cwd
        
        #eai job exec 05f24da2-18be-4174-bc21-475cdd7fd54c -- bash -c 'ls -l'

        cmd = ["eai", "job", "exec", self.job_id, "--", "bash", "-c"]
        env_var_cmd = []
        for key in self.config.forward_env:
            if (value := os.getenv(key)) is not None:
                env_var_cmd.extend([f"export {key}={value}"])
        for key, value in self.config.env.items():
            env_var_cmd.extend([f"export {key}={value}"])
        cmd.extend([" && ".join(env_var_cmd) + " && " + command])
        
        result = subprocess.run(
            cmd,
            text=True,
            timeout=timeout or self.config.timeout,
            encoding="utf-8",
            errors="replace",
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
        # print(f"EAI Command executed: {shlex.join(cmd)}")
        return {"output": result.stdout, "returncode": result.returncode}

    def cleanup(self):
        """Stop and remove the Docker container."""
        if getattr(self, "job_id", None) is not None:  # if init fails early, container_id might not be set
            cmd = f"eai job kill {self.job_id}"
            subprocess.Popen(cmd, shell=True)

    def __del__(self):
        """Cleanup container when object is destroyed."""
        self.cleanup()

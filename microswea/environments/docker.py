import shlex
import subprocess
import uuid
from dataclasses import dataclass, field
from typing import Any


@dataclass
class DockerEnvironmentConfig:
    image: str
    cwd: str = "/"
    env: dict[str, str] = field(default_factory=dict)
    timeout: int = 30


class DockerEnvironment:
    def __init__(self, **kwargs):
        """This class executes bash commands in a Docker container using direct docker commands."""
        self.container_id: str | None = None
        self.config = DockerEnvironmentConfig(**kwargs)
        self._start_container()

    def _start_container(self):
        """Start the Docker container and return the container ID."""
        container_name = f"microswea-{uuid.uuid4().hex[:8]}"
        cmd = [
            "docker",
            "run",
            "-d",
            "--name",
            container_name,
            "-w",
            "/testbed",
            self.config.image,
            "sleep",
            "infinity",  # Keep container running
        ]
        print(f"Starting container with command: {shlex.join(cmd)}")
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=120,  # docker pull might take a while
            check=True,
        )
        print(f"Started container {container_name} with ID {result.stdout.strip()}")
        self.container_id = result.stdout.strip()

    def execute(self, command: str, cwd: str = "") -> dict[str, Any]:
        """Execute a command in the Docker container and return the result as a dict."""
        cwd = cwd or self.config.cwd
        assert self.container_id, "Container not started"

        cmd = ["docker", "exec", "-w", cwd]
        for key, value in self.config.env.items():
            cmd.extend(["-e", f"{key}={value}"])
        cmd.extend([self.container_id, "bash", "-c", command])

        return vars(
            subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=self.config.timeout,
            )
        )

    def cleanup(self):
        """Stop and remove the Docker container."""
        if getattr(self, "container_id", None) is not None:  # if init fails early, container_id might not be set
            print(f"Stopping container {self.container_id} (might take a second)")
            subprocess.run(["docker", "stop", self.container_id], capture_output=True, check=False)  # type: ignore
            subprocess.run(["docker", "rm", self.container_id], capture_output=True, check=False)  # type: ignore

    def __del__(self):
        """Cleanup container when object is destroyed."""
        self.cleanup()

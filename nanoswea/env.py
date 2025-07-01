import shlex
import subprocess
import uuid


class LocalEnvironment:
    def __init__(self):
        """This class executes bash commands directly on the local machine."""
        pass

    def execute(self, command: str, cwd: str = "/testbed") -> str:
        """Execute a command in the local environment and return the raw output."""
        result = subprocess.run(
            command,
            shell=True,
            capture_output=True,
            text=True,
            cwd=cwd,
            timeout=30,
        )
        return f"<stdout>\n{result.stdout}\n</stdout>\n<stderr>\n{result.stderr}\n</stderr>\n<exit_code>\n{result.returncode}\n</exit_code>"


class DockerEnvironment:
    def __init__(self, image: str):
        """This class executes bash commands in a Docker container using direct docker commands."""
        self.container_id = None
        self._start_container(image)

    def _start_container(self, image: str):
        """Start the Docker container and return the container ID."""
        container_name = f"nanoswea-{uuid.uuid4().hex[:8]}"
        cmd = [
            "docker",
            "run",
            "-d",
            "--name",
            container_name,
            "-w",
            "/testbed",
            image,
            "sleep",
            "infinity",  # Keep container running
        ]
        print(f"Starting container with command: {shlex.join(cmd)}")
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=30,
            check=True,
        )
        print(f"Started container {container_name} with ID {result.stdout.strip()}")
        self.container_id = result.stdout.strip()

    def execute(self, command: str, cwd: str = "/testbed") -> str:
        """Execute a command in the Docker container and return the raw output."""
        if not self.container_id:
            msg = "Container not started"
            raise RuntimeError(msg)

        result = subprocess.run(
            ["docker", "exec", "-w", cwd, self.container_id, "bash", "-c", command],
            capture_output=True,
            text=True,
            timeout=30,
        )
        return f"<stdout>\n{result.stdout}\n</stdout>\n<stderr>\n{result.stderr}\n</stderr>\n<exit_code>\n{result.returncode}\n</exit_code>"

    def cleanup(self):
        """Stop and remove the Docker container."""
        if self.container_id:
            subprocess.run(["docker", "stop", self.container_id], capture_output=True, check=False)
            subprocess.run(["docker", "rm", self.container_id], capture_output=True, check=False)

    def __del__(self):
        """Cleanup container when object is destroyed."""
        self.cleanup()

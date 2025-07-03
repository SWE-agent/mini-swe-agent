import asyncio

from swerex.deployment.docker import DockerDeployment
from swerex.runtime.abstract import Command as RexCommand


class SwerexDockerEnvironment:
    def __init__(self, image: str):
        """This class executes bash commands in a Docker container using SWE-ReX for sandboxing."""
        self.deployment = DockerDeployment(image=image)
        asyncio.run(self.deployment.start())

    def execute(self, command: str, cwd: str = "/testbed") -> str:
        """Execute a command in the environment and return the raw output."""
        output = asyncio.run(
            self.deployment.runtime.execute(RexCommand(command=command, shell=True, check=False, cwd=cwd))
        )
        return f"stdout: {output.stdout}\nstderr: {output.stderr}\nexit_code: {output.exit_code}"

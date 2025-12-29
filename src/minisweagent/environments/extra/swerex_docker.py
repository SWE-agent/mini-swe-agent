import asyncio
import time
from typing import Any

from jinja2 import StrictUndefined, Template
from pydantic import BaseModel
from swerex.deployment.docker import DockerDeployment
from swerex.runtime.abstract import Command as RexCommand

from minisweagent.exceptions import ExecutionTimeoutError, Submitted
from minisweagent.utils.serialize import recursive_merge


class SwerexDockerEnvironmentConfig(BaseModel):
    image: str
    cwd: str = "/"
    """Working directory in which to execute commands."""
    timeout: int = 30
    """Timeout for executing commands in the container."""
    deployment_extra_kwargs: dict[str, Any] = {}
    """Extra kwargs to pass to DockerDeployment."""
    action_observation_template: str = (
        "<returncode>{{output.returncode}}</returncode>\n<output>\n{{output.output}}</output>"
    )
    """Template used to render the observation after executing an action."""
    timeout_template: str = "Command timed out. Output:\n{{output}}"
    """Template used when a command timed out."""


class SwerexDockerEnvironment:
    def __init__(self, **kwargs):
        """This class executes bash commands in a Docker container using SWE-ReX for sandboxing."""
        self.config = SwerexDockerEnvironmentConfig(**kwargs)
        self.deployment = DockerDeployment(image=self.config.image, **self.config.deployment_extra_kwargs)
        asyncio.run(self.deployment.start())

    def execute(self, command: str, cwd: str = "", *, timeout: int | None = None) -> dict[str, Any]:
        """Execute a command in the environment and return the raw output."""
        output = asyncio.run(
            self.deployment.runtime.execute(
                RexCommand(
                    command=command,
                    shell=True,
                    check=False,
                    cwd=cwd or self.config.cwd,
                    timeout=timeout or self.config.timeout,
                    merge_output_streams=True,
                )
            )
        )
        return {
            "output": output.stdout,
            "returncode": output.exit_code,
        }

    def execute_messages(self, messages: list[dict], extra_template_vars: dict[str, Any] | None = None) -> list[dict]:
        """Execute all actions in messages and return observation messages."""
        results = []
        for msg in messages:
            if "action" not in msg:
                continue
            try:
                output = self.execute(msg["action"])
            except (TimeoutError, asyncio.TimeoutError) as e:
                output_text = str(e) if e else ""
                raise ExecutionTimeoutError(
                    Template(self.config.timeout_template, undefined=StrictUndefined).render(
                        **self.get_template_vars({"action": msg["action"], "output": output_text}, extra_template_vars)
                    )
                )
            self.check_finished(output)
            results.extend(self.format_observation(msg, output))
        return results

    def format_observation(self, msg: dict, output: dict) -> list[dict]:
        """Format output as observation message(s)."""
        content = Template(self.config.action_observation_template, undefined=StrictUndefined).render(
            **self.get_template_vars({"action": msg["action"], "output": output})
        )
        return [{"role": "user", "content": content, "timestamp": time.time(), "extra": output}]

    def check_finished(self, output: dict):
        """Raises Submitted exception if the output indicates task completion."""
        lines = output.get("output", "").lstrip().splitlines(keepends=True)
        if lines and lines[0].strip() == "COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT":
            raise Submitted("".join(lines[1:]))

    def get_template_vars(self, *extra_dicts: dict[str, Any] | None) -> dict[str, Any]:
        return recursive_merge(self.config.model_dump(), *extra_dicts)

    def serialize(self) -> dict:
        return {
            "info": {
                "config": {
                    "environment": self.config.model_dump(mode="json"),
                    "environment_type": f"{self.__class__.__module__}.{self.__class__.__name__}",
                }
            }
        }

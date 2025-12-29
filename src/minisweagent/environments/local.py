import os
import platform
import subprocess
import time
from typing import Any

from jinja2 import StrictUndefined, Template
from pydantic import BaseModel

from minisweagent.exceptions import ExecutionTimeoutError, Submitted
from minisweagent.utils.serialize import recursive_merge


class LocalEnvironmentConfig(BaseModel):
    cwd: str = ""
    env: dict[str, str] = {}
    timeout: int = 30
    action_observation_template: str = (
        "<returncode>{{output.returncode}}</returncode>\n<output>\n{{output.output}}</output>"
    )
    """Template used to render the observation after executing an action."""
    timeout_template: str = "Command timed out. Output:\n{{output}}"
    """Template used when a command timed out."""


class LocalEnvironment:
    def __init__(self, *, config_class: type = LocalEnvironmentConfig, **kwargs):
        """This class executes bash commands directly on the local machine."""
        self.config = config_class(**kwargs)

    def execute(self, command: str, cwd: str = "", *, timeout: int | None = None):
        """Execute a command in the local environment and return the result as a dict."""
        cwd = cwd or self.config.cwd or os.getcwd()
        result = subprocess.run(
            command,
            shell=True,
            text=True,
            cwd=cwd,
            env=os.environ | self.config.env,
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
            if "action" not in msg:
                continue
            try:
                output = self.execute(msg["action"])
            except (TimeoutError, subprocess.TimeoutExpired) as e:
                output_text = e.output.decode("utf-8", errors="replace") if getattr(e, "output", None) else ""
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
        return recursive_merge(self.config.model_dump(), platform.uname()._asdict(), os.environ, *extra_dicts)

    def serialize(self) -> dict:
        return {
            "info": {
                "config": {
                    "environment": self.config.model_dump(mode="json"),
                    "environment_type": f"{self.__class__.__module__}.{self.__class__.__name__}",
                }
            }
        }

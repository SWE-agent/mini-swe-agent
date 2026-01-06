import os
import platform
import subprocess
import time
from typing import Any

from jinja2 import StrictUndefined, Template
from pydantic import BaseModel

from minisweagent.exceptions import Submitted
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

    def execute_messages(
        self, messages: list[dict], *, extra_template_vars: dict[str, Any] | None = None
    ) -> list[dict]:
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
                results.append(
                    {
                        "role": "user",
                        "content": Template(self.config.timeout_template, undefined=StrictUndefined).render(
                            **self.get_template_vars(action=action, output=output_text, **(extra_template_vars or {}))
                        ),
                        "extra": {"interrupt_type": "ExecutionTimeoutError", "timestamp": time.time()},
                    }
                )
                continue
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
        """Raises Submitted if the output indicates task completion."""
        lines = output.get("output", "").lstrip().splitlines(keepends=True)
        if lines and lines[0].strip() == "COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT":
            submission = "".join(lines[1:])
            raise Submitted(
                {
                    "role": "exit",
                    "content": submission,
                    "extra": {"exit_status": "Submitted", "submission": submission},
                }
            )

    def get_template_vars(self, **kwargs) -> dict[str, Any]:
        return recursive_merge(self.config.model_dump(), platform.uname()._asdict(), os.environ, kwargs)

    def serialize(self) -> dict:
        return {
            "info": {
                "config": {
                    "environment": self.config.model_dump(mode="json"),
                    "environment_type": f"{self.__class__.__module__}.{self.__class__.__name__}",
                }
            }
        }

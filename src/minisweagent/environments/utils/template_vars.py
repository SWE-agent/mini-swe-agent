import json
from dataclasses import asdict
from typing import Any

from minisweagent import Environment


def get_remote_template_vars(env: Environment) -> dict[str, Any]:
    """Get template variables (env variables etc.) from remote environments."""
    try:
        platform_info = json.loads(
            env.execute("python -c 'import platform; print(platform.uname()._asdict())'")["output"]
        )
    except ValueError:
        platform_info = {}
    env_output = env.execute("env")
    if env_output["returncode"] == 0:
        env_vars = dict([line.split("=", 1) for line in env_output["output"].splitlines()])
    else:
        env_vars = {}
    return platform_info | asdict(env.config) | env_vars

import json
import traceback
from dataclasses import asdict
from typing import Any

from minisweagent import Environment

_PLATFORM_INFO_COMMAND = """python -c '
import platform;
import json
print(json.dumps(platform.uname()._asdict()))
'
"""
_ENV_COMMAND = "env"


def get_remote_template_vars(env: Environment, strict: bool = False) -> dict[str, Any]:
    """Get template variables (env variables etc.) from remote environments."""
    platform_info_output = env.execute(_PLATFORM_INFO_COMMAND)
    try:
        platform_info = json.loads(platform_info_output["output"].strip())
    except ValueError:
        if strict:
            raise
        print(f"Failed to get platform info: {platform_info_output}\n{traceback.format_exc()}")
        platform_info = {}
    env_output = env.execute(_ENV_COMMAND)
    if env_output["returncode"] == 0:
        try:
            env_vars = dict([line.split("=", 1) for line in env_output["output"].splitlines()])
        except ValueError:
            if strict:
                raise
            print(f"Failed to get env vars: {env_output}\n{traceback.format_exc()}")
            env_vars = {}
    else:
        if strict:
            raise ValueError(f"Failed to get env vars: {env_output}")
        print(f"Failed to get env vars: {env_output}")
        env_vars = {}
    return platform_info | asdict(env.config) | env_vars

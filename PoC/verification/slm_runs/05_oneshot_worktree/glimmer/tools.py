"""Tools and their registry."""

import subprocess

import observatory_log


def bash(command: str) -> str:
    return subprocess.run(command, shell=True, capture_output=True, text=True, errors="replace").stdout


def python(code: str) -> str:
    scope: dict = {}
    exec(code, scope)
    return str(scope.get("result", ""))


class Registry:
    def __init__(self):
        self.tools = {"bash": bash, "python": python}

    def run(self, name: str, args: dict) -> str:
        start = observatory_log.tool_start(name, args)
        try:
            out = self.tools[name](**args)
        except Exception as e:
            observatory_log.tool_error(start, name, e)
            raise
        observatory_log.tool_end(start, name, result=out)
        return out

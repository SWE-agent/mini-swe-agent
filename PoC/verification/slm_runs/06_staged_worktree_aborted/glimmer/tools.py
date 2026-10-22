"""Tools and their registry."""

import subprocess



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
        out = self.tools[name](**args)
        return out

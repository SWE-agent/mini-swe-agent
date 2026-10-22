import subprocess
import sys
from pathlib import Path

repo_root = Path(
    "C:/Users/USER/AppData/Local/Temp/claude/c--Users-USER-Developments-mini-swe-agent/2b6776a6-acde-42c1-9433-2f8ca7ccb7f5/scratchpad/wt/glimmer"
)
archify_dir = Path(
    "C:/Users/USER/AppData/Local/Temp/claude/c--Users-USER-Developments-mini-swe-agent/2b6776a6-acde-42c1-9433-2f8ca7ccb7f5/scratchpad/archify-3.0.1/archify"
)
spec = Path(
    "C:/Users/USER/AppData/Local/Temp/claude/c--Users-USER-Developments-mini-swe-agent/2b6776a6-acde-42c1-9433-2f8ca7ccb7f5/scratchpad/wt/glimmer/PoC/verification/toy_agent/observatory/diagram/agent-architecture.json"
)

cmd = [
    "node",
    str(archify_dir / "bin/archify.mjs"),
    "validate",
    "architecture",
    str(spec),
    "--quality",
    "showcase",
    "--repo-root",
    str(repo_root),
]

print("Running:", " ".join(cmd))
result = subprocess.run(cmd, capture_output=True, text=True)
print("stdout:", result.stdout)
print("stderr:", result.stderr)
print("returncode:", result.returncode)
sys.exit(result.returncode)

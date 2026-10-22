import sys

sys.path.insert(0, ".")
import json

from observatory.flow_verify import Verifier

root = "C:/Users/USER/AppData/Local/Temp/claude/c--Users-USER-Developments-mini-swe-agent/2b6776a6-acde-42c1-9433-2f8ca7ccb7f5/scratchpad/wt/nemotron/PoC/verification/toy_agent"
anchors_path = "C:/Users/USER/AppData/Local/Temp/claude/c--Users-USER-Developments-mini-swe-agent/2b6776a6-acde-42c1-9433-2f8ca7ccb7f5/scratchpad/wt/nemotron/PoC/verification/toy_agent/observatory/diagram/verify-anchors.json"
output_dir = "C:/Users/USER/AppData/Local/Temp/claude/c--Users-USER-Developments-mini-swe-agent/2b6776a6-acde-42c1-9433-2f8ca7ccb7f5/scratchpad/wt/nemotron/PoC/verification/toy_agent/observatory/verify"

with open(anchors_path, encoding="utf-8") as f:
    anchors = json.load(f)

files = [root + "/agent.py", root + "/llm.py", root + "/tools.py"]

verifier = Verifier(root, files, anchors)

# Run verification
verifier.entry()
for q in verifier.funcs:
    verifier.enter(q)
    verifier.stmts(verifier.funcs[q]["node"].body)

# Print results
print("=== Verification Results ===")
for key in sorted(verifier.partial):
    print(f"{key}: {STATUS.get(key, 'unknown')}")

# Check if all are covered
all_covered = all(verifier.partial.get(k, "bypass") == "검증 완료" for k in verifier.partial)

print(f"\nAll covered: {all_covered}")

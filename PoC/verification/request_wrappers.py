# /// script
# requires-python = ">=3.10"
# dependencies = []
# ///
"""Send a wrapper request (wrap_request.md, made in the io_audit viewer) to Claude Code the way the kit does.

Uses agent_observatory_kit's own run.py (<repo>/observatory/run.py): the same Claude Code slot (memoryless session,
file edits + `uv run` + `git diff` only, no commit, budget cap), then the kit's wiring check.

usage (repo root):
  uv run --python 3.12 PoC/verification/request_wrappers.py wrap_request.md [--model sonnet] [--budget 3]
Review the result with `git diff`, then re-run io_audit.py to see the items covered.
"""

import argparse
import importlib.util
import os
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument("request", type=Path)
p.add_argument("--kit", type=Path, default=Path("observatory"), help="the installed kit (default: ./observatory)")
p.add_argument("--model", default="sonnet", help="Claude Code model (the kit's default: sonnet)")
p.add_argument("--budget", type=float, default=3.0, help="max USD for the session")
a = p.parse_args()

os.environ.setdefault("PYTHONUTF8", "1")  # the kit reads its files with the platform encoding (cp949 on Korean Windows)
spec = importlib.util.spec_from_file_location("observatory_run", a.kit.resolve() / "run.py")
kit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(kit)

kit.step(f"Claude Code ({a.model}, max ${a.budget}): {a.request}")
kit.claude(a.request.read_text(encoding="utf-8"), a.model, a.budget, None)
kit.step("Checking the result")
kit.check_wiring()
print("\nReview with `git diff`, then re-run io_audit.py (command at the end of the request).")

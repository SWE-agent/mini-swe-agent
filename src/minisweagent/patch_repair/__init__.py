"""Coder–Reviewer Patch Repair loop for mini-swe-agent.

When the agent's patch fails to apply or pass tests, feed the error trace
back to the LM and ask for a corrected patch (up to 2 retries).
"""

from minisweagent.patch_repair.repair import attempt_patch_repair

__all__ = ["attempt_patch_repair"]

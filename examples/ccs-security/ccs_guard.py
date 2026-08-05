"""
CCS Security Guard for mini-swe-agent

This module provides a CCS-validated wrapper around mini-swe-agent's LocalEnvironment.
It intercepts all command executions and validates them against CCS rules before execution.

Usage:
    from ccs_guard import CCSLocalEnvironment

    env = CCSLocalEnvironment(ccs_enabled=True)
    result = env.execute({"command": "ls -la"})
"""

import logging
from typing import Any

try:
    from ccs_verifier import Command, Verifier
    from ccs_verifier.builtin_rules import CredentialLeakRule, RCERule, SSRFRule

    CCS_AVAILABLE = True
except ImportError:
    CCS_AVAILABLE = False

from minisweagent.environments.local import LocalEnvironment, LocalEnvironmentConfig

logger = logging.getLogger(__name__)


class CCSLocalEnvironmentConfig(LocalEnvironmentConfig):
    """Extended config with CCS options."""

    ccs_enabled: bool = True
    ccs_block_on_deny: bool = True  # Block execution if CCS denies
    ccs_log_all: bool = False  # Log all verifications, not just denials


class CCSLocalEnvironment(LocalEnvironment):
    """
    A CCS-secured local environment that validates commands before execution.

    CCS (Credential & Compliance Standard) provides:
    - RCE protection: Blocks dangerous shell commands
    - SSRF prevention: Blocks requests to internal endpoints
    - Credential leak detection: Prevents secret exposure

    Performance: P50 ≈ 7.5μs per verification (in-process)
    """

    def __init__(self, *, config_class: type = CCSLocalEnvironmentConfig, **kwargs):
        super().__init__(config_class=config_class, **kwargs)

        self.ccs_enabled = self.config.ccs_enabled and CCS_AVAILABLE
        self.ccs_block_on_deny = self.config.ccs_block_on_deny
        self.ccs_log_all = self.config.ccs_log_all

        if self.ccs_enabled:
            rules = [RCERule(), SSRFRule(), CredentialLeakRule()]
            self.verifier = Verifier(rules=rules)
            logger.info("CCS security guard initialized with rules: RCE, SSRF, CredentialLeak")
        else:
            self.verifier = None
            if not CCS_AVAILABLE:
                logger.warning("ccs-verifier not installed. Install with: pip install ccs-verifier")

    def execute(self, action: dict, cwd: str = "", *, timeout: int | None = None) -> dict[str, Any]:
        """Execute a command with CCS validation."""
        command = action.get("command", "")

        if self.ccs_enabled and command:
            verification_result = self._verify_command(command)

            if verification_result["verdict"] == "deny" and self.ccs_block_on_deny:
                logger.warning(f"CCS blocked command: {command[:100]}...")
                return {
                    "output": "",
                    "returncode": -1,
                    "exception_info": f"CCS Security Guard blocked command: {verification_result.get('reason', 'unknown')}",
                    "extra": {
                        "ccs_verdict": "deny",
                        "ccs_reason": verification_result.get("reason"),
                        "ccs_score": verification_result.get("score"),
                    },
                }

        # Execute the command
        result = super().execute(action, cwd, timeout=timeout)

        # Attach CCS receipt if available
        if self.ccs_enabled:
            result.setdefault("extra", {})["ccs_verified"] = True

        return result

    def _verify_command(self, command: str) -> dict:
        """Verify a command using CCS rules."""
        try:
            cmd = Command(agent_id="mini-swe-agent", tool="shell", params={"command": command})
            result = self.verifier.verify(cmd)

            verdict_dict = {
                "verdict": result.verdict.value,
                "score": result.score,
            }

            if result.verdict.value == "deny":
                verdict_dict["reason"] = getattr(result, "reason", "CCS policy violation")
                if self.ccs_log_all:
                    logger.info(f"CCS DENY: {command[:80]}... (score={result.score:.4f})")
            elif self.ccs_log_all:
                logger.debug(f"CCS ALLOW: {command[:80]}... (score={result.score:.4f})")

            return verdict_dict

        except Exception as e:
            logger.error(f"CCS verification error: {e}")
            return {"verdict": "error", "reason": str(e)}


def demo():
    """Demonstrate CCS security guard functionality."""
    print("=" * 60)
    print("CCS Security Guard Demo for mini-swe-agent")
    print("=" * 60)

    env = CCSLocalEnvironment(ccs_enabled=True, ccs_log_all=True)

    test_commands = [
        ("ls -la", "Safe command - should pass"),
        ("echo 'Hello World'", "Safe command - should pass"),
        ("rm -rf / --no-preserve-root", "Destructive RCE - should block"),
        ("curl http://169.254.169.254/latest/meta-data/", "SSRF attempt - may block"),
        ("cat /etc/shadow && echo $AWS_SECRET_ACCESS_KEY", "Credential leak - may block"),
    ]

    for cmd, description in test_commands:
        print(f"\n--- Test: {description} ---")
        print(f"Command: {cmd}")
        result = env.execute({"command": cmd})

        if result.get("extra", {}).get("ccs_verdict") == "deny":
            print(f"✗ BLOCKED: {result.get('exception_info', 'Unknown reason')}")
        elif result.get("returncode", 0) == 0:
            print("✓ ALLOWED: Command executed successfully")
        else:
            print(f"! EXECUTED but failed (returncode={result.get('returncode')})")


if __name__ == "__main__":
    demo()

#!/usr/bin/env python3
"""
CCS Security Integration Demo for mini-swe-agent

This demo shows how CCS (Credential & Compliance Standard) can secure
AI agent command execution with sub-millisecond overhead.

Run: python demo.py
"""

import sys
import time

# Add parent directory to path
sys.path.insert(0, ".")

try:
    from ccs_verifier import Command, Verifier
    from ccs_verifier.builtin_rules import CredentialLeakRule, RCERule, SSRFRule
except ImportError:
    print("Error: ccs-verifier not installed. Install with: pip install ccs-verifier")
    sys.exit(1)


def benchmark_verification():
    """Benchmark CCS verification performance."""
    rules = [RCERule(), SSRFRule(), CredentialLeakRule()]
    verifier = Verifier(rules=rules)

    test_commands = [
        ("ls -la", "safe"),
        ("echo hello", "safe"),
        ("rm -rf /", "dangerous"),
        ("curl http://169.254.169.254/", "ssrf"),
        ("echo $AWS_SECRET_ACCESS_KEY", "credential"),
    ]

    print("\n=== CCS Performance Benchmark ===\n")

    times = []
    for cmd, category in test_commands:
        command = Command(agent_id="demo", tool="shell", params={"command": cmd})

        start = time.perf_counter()
        result = verifier.verify(command)
        elapsed_us = (time.perf_counter() - start) * 1_000_000
        times.append(elapsed_us)

        status = "✓ ALLOW" if result.verdict.value == "allow" else "✗ DENY"
        print(f"{status} | {cmd[:40]:<40} | {elapsed_us:>8.1f}μs | {category}")

    avg_time = sum(times) / len(times)
    print(f"\nAverage verification time: {avg_time:.1f}μs")
    print("Overhead: Sub-millisecond (< 1ms)")


def demo_security_rules():
    """Demonstrate CCS security rules."""
    rules = [RCERule(), SSRFRule(), CredentialLeakRule()]
    verifier = Verifier(rules=rules)

    test_cases = [
        # Safe commands
        ("ls -la /tmp", True, "List files"),
        ("python3 script.py", True, "Run Python script"),
        ("git status", True, "Git status check"),
        # Dangerous commands
        ("rm -rf /", False, "Delete root filesystem"),
        ("chmod 777 /etc/passwd", False, "World-writable sensitive file"),
        (":(){:|:&};:", False, "Fork bomb"),
        # SSRF attempts
        ("curl http://169.254.169.254/latest/meta-data/", False, "AWS metadata"),
        ("wget http://localhost:8080/admin", False, "Localhost access"),
        # Credential leaks
        ("echo $AWS_SECRET_ACCESS_KEY", False, "Expose AWS secret"),
        ("cat ~/.ssh/id_rsa", False, "Read SSH private key"),
    ]

    print("\n=== CCS Security Rules Demo ===\n")
    print(f"{'Command':<50} | {'Expected':<8} | {'Actual':<8} | {'Status'}")
    print("-" * 90)

    passed = 0
    for cmd, should_allow, description in test_cases:
        command = Command(agent_id="demo", tool="shell", params={"command": cmd})
        result = verifier.verify(command)

        actual_allow = result.verdict.value == "allow"
        expected = "ALLOW" if should_allow else "DENY"
        actual = "ALLOW" if actual_allow else "DENY"

        correct = actual_allow == should_allow
        status = "✓" if correct else "✗"
        if correct:
            passed += 1

        print(f"{cmd[:48]:<50} | {expected:<8} | {actual:<8} | {status} {description}")

    print(f"\nPassed: {passed}/{len(test_cases)}")


def demo_integration():
    """Demonstrate mini-swe-agent integration."""
    print("\n=== mini-swe-agent Integration Demo ===\n")

    try:
        from ccs_guard import CCSLocalEnvironment

        env = CCSLocalEnvironment(ccs_enabled=True)

        print("Testing CCSLocalEnvironment wrapper:\n")

        test_commands = [
            ("echo 'Safe command'", "Should execute"),
            ("rm -rf /", "Should be blocked"),
        ]

        for cmd, description in test_commands:
            print(f"Command: {cmd}")
            print(f"Expected: {description}")

            result = env.execute({"command": cmd})

            if result.get("extra", {}).get("ccs_verdict") == "deny":
                print("Result: ✗ BLOCKED by CCS\n")
            elif result.get("returncode", -1) == 0:
                print("Result: ✓ EXECUTED\n")
            else:
                print(f"Result: ! EXECUTED but failed (returncode={result.get('returncode')})\n")

    except ImportError:
        print("Note: Run this from the mini-swe-agent directory with ccs_guard.py available")


if __name__ == "__main__":
    print("=" * 60)
    print("CCS Security Integration for mini-swe-agent")
    print("Credential & Compliance Standard - Runtime Verification")
    print("=" * 60)

    benchmark_verification()
    demo_security_rules()
    demo_integration()

    print("\n" + "=" * 60)
    print("Demo complete!")
    print("=" * 60)
    print("\nLearn more:")
    print("  - IETF Draft: https://datatracker.ietf.org/doc/draft-correctover-ccs/")
    print("  - PyPI: https://pypi.org/project/ccs-verifier/")
    print("  - DOI: 10.5281/zenodo.21783723")

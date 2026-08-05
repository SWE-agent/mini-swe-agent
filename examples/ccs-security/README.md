# CCS Security Integration for mini-swe-agent

This example demonstrates how to integrate [CCS (Credential & Compliance Standard)](https://github.com/Correctover/ccs-verifier) 
into mini-swe-agent for runtime command verification.

## What is CCS?

CCS is an IETF-standardized runtime verification framework that provides:
- **RCE Protection**: Detects dangerous shell commands (rm -rf, chmod 777, etc.)
- **SSRF Prevention**: Blocks requests to internal/cloud metadata endpoints
- **Credential Leak Detection**: Prevents exposure of secrets and API keys
- **Sub-millisecond Overhead**: P50 ≈ 7.5μs in-process verification

## Quick Start

```python
from ccs_verifier import Verifier, Command
from ccs_verifier.builtin_rules import RCERule, SSRFRule, CredentialLeakRule

# Initialize verifier with built-in rules
rules = [RCERule(), SSRFRule(), CredentialLeakRule()]
verifier = Verifier(rules=rules)

# Verify command before execution
cmd = Command(agent_id="mini-swe-agent", tool="shell", params={"command": "ls -la"})
result = verifier.verify(cmd)

if result.verdict.value == "allow":
    # Safe to execute
    execute_command(cmd.params["command"])
else:
    # Block dangerous command
    print(f"Command blocked: {result.reason}")
```

## Integration with mini-swe-agent

See `ccs_guard.py` for a complete implementation that wraps mini-swe-agent's 
`LocalEnvironment` with CCS validation.

## References

- [CCS IETF Draft](https://datatracker.ietf.org/doc/draft-correctover-ccs/)
- [CCS PyPI Package](https://pypi.org/project/ccs-verifier/)
- [Zenodo DOI: 10.5281/zenodo.21783723](https://doi.org/10.5281/zenodo.21783723)

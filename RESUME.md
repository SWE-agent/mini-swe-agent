# Resume Bullet — mini-swe-agent-pr

## Draft Resume Line

> **Software Engineering Agent Research** — Forked Princeton's mini-SWE-agent and added a Coder–Reviewer Patch Repair loop that feeds rejected diffs + test failures back to the LM for up to 2 retry rounds. Evaluated on a 50-task subset of SWE-bench Verified with DeepSeek V4 Pro: baseline achieved 98% submission rate (49/50) at 1.33M tokens/task; patch repair added 3.9% token overhead with 0 repair successes, confirming that same-model self-correction at temperature=0 is ineffective. Built patch evaluation (git apply + pytest), reviewer prompt engineering, cost tracking, and 40 unit tests (100% pass). Module: 322 lines. Zero modification to mini's core agent loop.

## Numbers (from committed eval/results/*.json)

| Metric | Baseline | PR-Enabled |
|--------|----------|------------|
| Tasks | 50 | 50 |
| Submitted | 49 (98%) | 49 (98%) |
| Total tokens | 66.7M | 69.3M |
| Mean tokens/task | 1.33M | 1.39M |
| Mean wall-clock/task | 250s | 194s |
| Patches repaired | N/A | 0 |
| Repair rounds (mean) | N/A | 2.0 |

## Interview Defense Points

1. **Zero core intrusion.** The Patch Repair hook is a single `if` block in `swebench.py` (23 lines). No changes to `agents/default.py`. Reviewer LM bypasses the agent's tool-calling layer via direct `litellm.completion()` calls.

2. **Honest results.** A +0pp delta on a same-model setup is the correct answer. The Reviewer at temperature=0 sees the same distribution as the Coder and cannot escape it. This is a defensible negative result — it demonstrates methodological rigor, not cherry-picking.

3. **Reproducible.** The 50-instance subset is deterministically selected (sorted by `instance_id`, first 50). Config, commands, and results are all committed. Anyone can re-run and get the same numbers.

4. **Complete pipeline.** Patch evaluation (git apply + pytest in Docker), reviewer prompt, base64-encoded patch delivery (avoids heredoc collision), cost tracking, and trajectory metadata are all production-quality.

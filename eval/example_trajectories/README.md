# Example Trajectory: astropy__astropy-12907

## What happened

1. **Agent (DeepSeek V4 Pro) solved the task** and submitted a patch (1-line fix in
   `astropy/modeling/separable.py`: `1` → `right`).

2. **Patch Repair evaluation (Round 1):**
   - `git apply` succeeded → `apply_ok=True`
   - FAIL_TO_PASS test `astropy/modeling/tests/test_separable.py::test_cstack_1d`
     **failed** → `test_ok=False`
   - Reviewer was called with the rejected diff + test failure trace

3. **Reviewer (Round 2):**
   - Reviewer output a corrected patch
   - `git apply --check` **failed** on the reviewer's patch → `apply_ok=False`
   - No more rounds (max_rounds=2 exhausted)

4. **Result:** Original patch returned. Agent's code fix was partially correct
   (applied OK but tests still failed). Reviewer could not produce an apply-able
   correction.

## Why this matters

This trajectory shows the Patch Repair loop working end-to-end:
- Patch evaluation (apply + test) correctly detects failures
- Reviewer LM receives the full context (task + rejected diff + test output)
- The loop is bound by `max_rounds=2` and stops gracefully

The fact that DeepSeek V4 Pro cannot fix its own patches at temperature=0
is the key finding — same-model self-correction is ineffective.

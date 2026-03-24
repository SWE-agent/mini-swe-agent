# PRD: Multiagent Reviewer System (SWE-bench)

## Motivation

Confidently incorrect submissions are a key failure mode in SWE-bench. The main agent can reason itself into a plausible-looking-but-wrong fix. An independent reviewer — with no knowledge of the main agent's reasoning — can catch logic errors, missing edge cases, and incorrect assumptions by approaching the problem fresh.

## Scope

This system is designed exclusively for SWE-bench evaluation runs. All config, prompts, and submission mechanics are SWE-bench-specific. SWE-bench instances run in parallel in isolated Docker containers; each instance has its own independent environment and `__CLAUDE.md__`.

## Goals

- Reduce confidently incorrect submissions via a post-completion reviewer agent
- Keep agents independent: shared factual context only, no shared solution reasoning
- Remain composable with the existing `DefaultAgent` architecture
- Be configurable: number of review iterations, per-role cost/step limits

---

## Design

### Agent Roles

**Main Agent**
- Receives the original SWE-bench PR description
- Works in `/testbed` (the SWE-bench Docker working directory)
- Writes factual discoveries to `/testbed/__CLAUDE.md__` before submitting
- On clean confident exit only (not `LimitsExceeded` or exception), hands off to the Reviewer
- Submission command (existing SWE-bench convention):
  ```bash
  echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT && cat patch.txt
  ```
  The orchestrator captures the patch content from stdout following the signal.

**Reviewer Agent**
- Receives the **original PR description** only — not the main agent's messages or reasoning
- Reads `/testbed/__CLAUDE.md__` for factual context
- Uses git to understand prior changes: `git log`, `git diff HEAD`, `git show`
- Reviews logic, writes new tests/scripts to verify correctness
- Makes fixes if issues are found; may revert and start over if the approach is fundamentally broken
- Writes its own factual discoveries to `/testbed/__CLAUDE.md__` before exiting
- Exits via one of:
  ```bash
  echo REVIEWER_APPROVED && cat patch.txt   # no issues found; patch.txt is the final submission
  echo REVIEWER_COMPLETE && cat patch.txt   # fixes applied; patch.txt is the updated submission
  ```
- `LimitsExceeded` or exception exit: treated as no-confidence; iteration cap is decremented

### Submission Mechanics

All agents produce a `patch.txt` in `/testbed` following the SWE-bench convention:
- `git diff -- <modified source files> > patch.txt` (source files only; no tests, scripts, or configs)
- The patch must have `--- a/` and `+++ b/` headers

The orchestrator captures the patch content from stdout after the exit signal. The **reviewer's patch takes precedence** over the main agent's if the reviewer ran and exited cleanly. If the reviewer exits with `REVIEWER_APPROVED`, the patch from that command (which reflects the main agent's unchanged work) is used.

### Shared Context: `/testbed/__CLAUDE.md__`

- Created fresh per SWE-bench instance (cleared at orchestrator startup, before the main agent runs)
- Appended to by each agent before exiting
- Contains only **factual, objective** information:
  - Where relevant source files and tests live
  - Commands that reproduce the issue or run the test suite
  - Observed input/output behavior of relevant functions
- **Must not contain:** fix hypotheses, root cause reasoning, or any reference to a chosen approach

### Completion Signals

| Exit command | Agent | Meaning | Triggers next agent? |
|---|---|---|---|
| `echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT && cat patch.txt` | Main | Confident, clean completion | Yes → Reviewer |
| `LimitsExceeded` / exception | Main | Hit limits or crashed | No |
| `echo REVIEWER_APPROVED && cat patch.txt` | Reviewer | Work is correct, no changes | Terminates chain |
| `echo REVIEWER_COMPLETE && cat patch.txt` | Reviewer | Fixes applied | Next iteration or terminate if at cap |
| `LimitsExceeded` / exception | Reviewer | Hit limits or crashed | Counts against cap; terminate |

---

## Orchestration: `MultiAgentOrchestrator`

A new top-level class (not a subclass of `DefaultAgent`) with its own config.

### Flow

```
1. Clear /testbed/__CLAUDE.md__
2. Run MainAgent(task)
   └─ exit != COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT → return failure (no review)
3. current_patch = patch from main agent stdout
4. for iteration in range(max_review_iterations):
   a. Run ReviewerAgent(task)  ← fresh instance, same Docker env, same /testbed
   b. exit == REVIEWER_APPROVED → return current_patch (main agent's or last reviewer's)
   c. exit == REVIEWER_COMPLETE → current_patch = reviewer's patch; continue loop
   d. LimitsExceeded / exception → break (use current_patch as-is)
5. Return current_patch
```

### Config

```yaml
# config/benchmarks/swebench_multiagent.yaml
orchestrator:
  max_review_iterations: 1

main_agent:
  # inherits from swebench.yaml agent section
  system_template: ...
  instance_template: ...   # includes __CLAUDE.md__ write instruction
  step_limit: 250
  cost_limit: 3.0

reviewer_agent:
  system_template: ...     # review-focused persona
  instance_template: ...   # includes git discovery + __CLAUDE.md__ read/write instruction
  step_limit: 150          # reviewer gets fewer steps by default
  cost_limit: 2.0
```

### Combined Trajectory

One merged JSON file per instance. Structure:

```json
{
  "info": {
    "model_stats": {
      "instance_cost": "<sum of all agents>",
      "api_calls": "<sum of all agents>"
    },
    "exit_status": "<final exit signal>",
    "submission": "<final patch content>",
    "n_review_iterations": 1
  },
  "agents": [
    { "role": "main", "messages": [...], "info": { ... } },
    { "role": "reviewer", "iteration": 0, "messages": [...], "info": { ... } }
  ],
  "trajectory_format": "mini-swe-agent-multiagent-1.0"
}
```

---

## File Structure

```
src/minisweagent/
  agents/
    default.py                  # unchanged
    reviewer.py                 # ReviewerAgent (thin DefaultAgent subclass)
    multi_agent.py              # MultiAgentOrchestrator
  config/
    benchmarks/
      swebench.yaml             # unchanged (main agent config)
      swebench_multiagent.yaml  # orchestrator + reviewer config (extends swebench.yaml)
  run/
    multi.py                    # CLI entry point for the multiagent flow
```

---

## Reviewer System Prompt (sketch)

```
You are an independent code reviewer working on a SWE-bench task.
A previous agent has already attempted a fix. You do NOT have access to their reasoning.

Your job:
1. Read /testbed/__CLAUDE.md__ for factual context about the codebase
2. Use git to understand what was changed:
     git log --oneline -5
     git diff HEAD          (or git diff if changes are unstaged)
3. Independently verify whether the fix is correct:
   - Read the changed source files carefully
   - Write new scripts or tests to probe edge cases (do NOT modify existing tests)
   - Run the existing test suite
4. If correct: write your factual findings to __CLAUDE.md__, then:
     echo REVIEWER_APPROVED && cat patch.txt
5. If issues exist: fix them (you may revert and start over), then regenerate patch.txt
   and write your factual findings to __CLAUDE.md__, then:
     echo REVIEWER_COMPLETE && cat patch.txt

Patch rules (same as main agent):
- git diff -- <modified source files only> > patch.txt
- Do NOT include test files, reproduction scripts, or configs in the patch
- Verify patch.txt has --- a/ and +++ b/ headers before submitting

Do not reference or replicate the previous agent's reasoning. Approach this fresh.
```

---

## Reviewer Instance Template (sketch)

```
<pr_description>
{{task}}
</pr_description>

<instructions>
You are reviewing a previous agent's attempted fix for the above PR.

Start by reading /testbed/__CLAUDE.md__ for any factual context left by the previous agent,
then use git to discover what was changed before forming any conclusions.

[... standard SWE-bench environment/command rules ...]

## Submission

When done reviewing (and fixing if needed), submit following these steps:

Step 1: Ensure patch.txt reflects the correct final state of source files only:
  git diff -- path/to/file1 path/to/file2 > patch.txt

Step 2: Append your factual findings to /testbed/__CLAUDE.md__

Step 3a: If the solution is correct (no changes needed):
  echo REVIEWER_APPROVED && cat patch.txt

Step 3b: If you made fixes:
  echo REVIEWER_COMPLETE && cat patch.txt
</instructions>
```

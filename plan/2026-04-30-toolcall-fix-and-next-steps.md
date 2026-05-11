# PlanMem on SWE-bench Verified — analysis & next steps (2026-04-30)

## Status (snapshot)

- 500-run: workers=4, in progress (~7 m for first 8 instances → ETA ~6 h total).
- vLLM: Qwen3-Coder-30B-A3B-Instruct-FP8 on GPU3, max_model_len 32768, with
  `--enable-auto-tool-choice --tool-call-parser qwen3_coder`.
- Smoke (1 instance): Submitted, patch=327 chars, 50 calls.
- 5-instance batch: 5/5 Submitted (patch lens 456–15232).

Baseline reference (vanilla DefaultAgent, same Qwen3-Coder, same 500-instance
SWE-bench Verified, recorded in `experiments/BASELINE_RECORD.md`):
**146 / 500 resolved = 29.2 %**.

## The dead-mechanism problem (predictable before sb-cli result lands)

To make the agent runnable in toolcall mode at all, `PlanMemAgent.query` now
keeps `self.messages` intact when toolcall ↔ tool pairing must be preserved.
That guard is correct (without it the agent stalled at 250 steps with empty
patches), but the side effect is severe:

| PlanMem mechanism                                    | Reaches the model in toolcall mode? |
|------------------------------------------------------|--------------------------------------|
| Repo background card (system-prompt edit)            | ✅                                   |
| Phase-adaptive params (budget, λ, w_content/w_graph) | ❌ computed, never applied to context |
| Priority-file boost                                  | ❌ added to selected_nodes only       |
| Goal-reminder injection                              | ❌ added to selected_nodes only       |
| Replan / sub-task tree                               | ❌ internal state only                |
| Phase / drift / memory→planner stats                 | ❌ logged only                        |

So in toolcall mode, **PlanMem ≈ DefaultAgent + repo background card + a lot
of dead telemetry**. Until we route the planner's signals into the actual
prompt, we cannot expect to beat the 29.2 % baseline by much.

## Three concrete fixes (in priority order)

### A. System-prompt-level planning annotation (cheapest, biggest win)

Append a small block to the system message on every step (already proven path
via the repo background card):

```
<!-- PLANMEM_HEADER:BEGIN -->
Current phase: {phase}
Active sub-task: {active_subtask.description}
Recent priority files: {comma-separated top 5}
{if drift} Goal reminder: {goal_summary}
{if backtrack} You appear stuck — switch strategy: {recovery hint}
<!-- PLANMEM_HEADER:END -->
```

Implementation: a new `_apply_planning_header()` method in `PlanMemAgent`,
called inside `query()` after `_ensure_repo_background_card()`. Idempotent —
strips the prior block and rewrites it on each step.

Estimated cost: small system-prompt growth (few hundred chars). Caching wins
remain because the prefix changes only when phase/active subtask flips.

### B. Pair-aware message dropping (toolcall-safe context truncation)

When the conversation is long, drop *whole tool-call pairs* (assistant turn +
its matching tool messages) rather than individual nodes. This preserves
pairing while still letting the beam search trim to budget.

Implementation: extend `construct_context_via_search` to operate on
"turn groups" instead of nodes:
- Group nodes by `tool_call_id` adjacency.
- Beam search picks groups (atomic).
- Reorder by `id` then emit.

Risk: long-tail correctness — one mis-paired turn = API error. Add a
post-filter that asserts pairing before sending.

### C. Adopt textbased model for ablation runs

`LitellmTextbasedModel` parses ```mswea_bash_command``` fences from the
assistant text — no toolcall pairing constraint. PlanMem's full context-
rewriting machinery works there. Useful as an **ablation control** even if we
ship toolcall as the production path.

Switch via `-c model.model_class=litellm_textbased`.

## Other suspected weaknesses (validate from trajectory analysis)

Run `python -m experiments.eval.analyze_trajectories --traj-dir <out>` after
the batch finishes. Things to check:

1. **`avg_metadata_hit_rate`** — should be > 0.6. Smoke saw only 2/50 (4 %).
   Probably because most actions are `find ... | grep ...` which our
   `_extract_filenames` doesn't handle. Extend the parser to grab path-shaped
   tokens from any pipeline.
2. **Phase histogram** — if exploration ≈ 100 %, our phase detector is
   under-triggering implementation/verification. Tighten `_is_implementation`
   to recognise toolcall-shaped commands (e.g. `python -c '...write...'`).
3. **`avg_planner_done` vs `avg_planner_total`** — if mean done ≈ 0,
   sub-task verification predicate is too strict. Loosen the EXPLORATION
   exit criterion or count tool messages with non-empty output as evidence.
4. **`avg_nonzero_rc_rate`** — high return-code failures correlate with
   replan firing. If replan rarely fires despite high failure rate, the
   `consecutive_failure_threshold` may be too high in toolcall mode where
   the model recovers locally.
5. **Patch-empty rate among Submitted exits** — submitting an empty patch
   counts as Submitted in mini-swe-agent semantics but resolves 0 %. Filter
   for these and check why the model thought it was done.

## What "30 %+" means for this run

Baseline: 29.2 %. PlanMem on the same model with most mechanisms inert:
expected ≤ 30 % (might be 28–31 % from noise). To meaningfully beat baseline
we likely need fix A (system-prompt-level annotation) implemented and re-run.

If the live 500-run lands at 28–31 %, that is **the diagnostic signal**: the
co-design hypothesis is sound but the toolcall plumbing is the bottleneck.
Implement fix A, re-run 50-instance subset, then full if delta is positive.

## Repro pointers

- Trajectories: `experiments/outputs/planmem_full/<id>/<id>.traj.json`
- preds: `experiments/outputs/planmem_full/preds.json`
- vLLM log: `experiments/logs/vllm_server.log`
- Run log: `experiments/logs/full_500.log`
- Submit script: `experiments/submit_and_check.sh`
- Analyzer: `experiments/eval/analyze_trajectories.py`
- Comparator: `experiments/eval/compare_to_baseline.py`

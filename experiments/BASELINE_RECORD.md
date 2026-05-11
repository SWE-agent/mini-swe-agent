# Qwen3-Coder-30B-A3B Baseline on SWE-bench Verified — Audit Record

Document for external audit of the mini-swe-agent v2.2.8 baseline run. Last updated 2026-04-22.

## Result (sb-cli authoritative)

- Final score: **146 / 500 resolved = 29.20%** on SWE-bench Verified
- Run ID on sb-cli: `qwen3_coder_baseline_rerun_complete`
- **Authoritative report** (latest, after sb-cli finished all async eval): [sb-cli-reports/swe-bench_verified__test__qwen3_coder_baseline_rerun_complete.json](../sb-cli-reports/swe-bench_verified__test__qwen3_coder_baseline_rerun_complete.json)
  - Status breakdown: resolved 146, unresolved 353, completed 499, pending 0, failed 1 (`sympy__sympy-15875` — sb-cli eval infra failure, patch was submitted and well-formed; `failed` here means sb-cli's own sandbox errored, not our agent)
- Stale earlier snapshot (taken while `sympy__sympy-15875` was still pending): [sb-cli-reports/Subset.swe_bench_verified__test__qwen3_coder_baseline_rerun_complete.json](../sb-cli-reports/Subset.swe_bench_verified__test__qwen3_coder_baseline_rerun_complete.json) — same 146 resolved, but shows `pending=1, failed=0`. Keep for provenance; ignore for scoring.

## Reproducibility

### Codebase
- Repo root: `/data/jiahao/code/mini-swe-agent`
- Commit: `58ad92fc1385be3f8f2326b7a42320bb3655e169` (`Bump version`, tagged v2.2.8)
- Installed package: `mini-swe-agent 2.2.8` (editable install in conda env `swe_agent`)
- Agent default config: [src/minisweagent/config/benchmarks/swebench.yaml](../src/minisweagent/config/benchmarks/swebench.yaml) — **unmodified** from upstream v2.2.8
  - System/instance templates, submit protocol (`echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT && cat patch.txt`), step_limit=250, cost_limit=3.0
  - environment_class=docker, timeout=60s, cwd=/testbed

### Model serving (vLLM on local GPUs 0,3)
```
conda activate vllm_qwen
HF_HOME=/data/jiahao/.cache/huggingface \
TRITON_CACHE_DIR=/data/jiahao/.cache/triton \
VLLM_CACHE_ROOT=/data/jiahao/.cache/vllm \
CUDA_VISIBLE_DEVICES=0,3 \
vllm serve Qwen/Qwen3-Coder-30B-A3B-Instruct-FP8 \
  --tensor-parallel-size 2 \
  --gpu-memory-utilization 0.65 \
  --max-model-len 65536 \
  --host 0.0.0.0 --port 8765 \
  --served-model-name Qwen/Qwen3-Coder-30B-A3B-Instruct-FP8 \
  --enable-auto-tool-choice \
  --tool-call-parser qwen3_coder
```
- Model: `Qwen/Qwen3-Coder-30B-A3B-Instruct-FP8` (Qwen's official FP8 release, MoE 30B-A3B)
- Context: 65536 tokens (vLLM max_model_len)
- Tool-call parser: `qwen3_coder` (vendor-provided parser, required for tool-calling with this model)
- vLLM logs: [logs/vllm_server.log](logs/vllm_server.log)

### Litellm registry (for LiteLLM metadata/cost only — not trimming)
File [registry.json](registry.json):
```json
{
  "Qwen/Qwen3-Coder-30B-A3B-Instruct-FP8": {
    "max_tokens": 262144,
    "max_input_tokens": 262144,
    "max_output_tokens": 65536,
    "input_cost_per_token": 0.0,
    "output_cost_per_token": 0.0,
    "litellm_provider": "hosted_vllm",
    "mode": "chat",
    "supports_function_calling": true
  },
  "hosted_vllm/Qwen/Qwen3-Coder-30B-A3B-Instruct-FP8": { "...same...": "..." }
}
```
Exported to the agent via `LITELLM_MODEL_REGISTRY_PATH`.

### Agent override config
File [qwen3_coder_baseline.yaml](qwen3_coder_baseline.yaml) — only overrides the `model:` section:
```yaml
model:
  model_name: "hosted_vllm/Qwen/Qwen3-Coder-30B-A3B-Instruct-FP8"
  model_kwargs:
    api_base: "http://localhost:8765/v1"
    api_key: "EMPTY"
    temperature: 0.0
    drop_params: true
    parallel_tool_calls: true
  litellm_model_registry: "/data/jiahao/code/mini-swe-agent/experiments/registry.json"
  cost_tracking: "ignore_errors"
```
Everything else (system prompt, instance template, step_limit, submit protocol) is inherited unchanged from upstream `swebench.yaml`.

### Run command
Invocation (same for both runs, only `--workers` differs):
```bash
mini-extra swebench --subset verified --split test \
  -c swebench -c experiments/qwen3_coder_baseline.yaml \
  --workers N \
  --output experiments/qwen3_baseline_verified
```
- Initial run wrapper: [run_verified.sh](run_verified.sh) — `--workers 4`, resume loop (60s sleep between docker crashes). `mini-extra swebench` skip logic reads `preds.json` and skips any instance already present ([src/minisweagent/run/benchmarks/swebench.py:243](../src/minisweagent/run/benchmarks/swebench.py#L243)), so restarts resume from where `preds.json` left off.
- Rerun wrapper: [run_rerun.sh](run_rerun.sh) — `--workers 3`, wrapped with `nice -n 19 ionice -c 3`, `ulimit -v 64G`, `ulimit -n 8192` (politeness on shared server with a 16-worker run from another tenant).

### Rerun preparation (required to reproduce)
Because `mini-extra swebench` only skips instances *present* in `preds.json`, to get the rerun to re-attempt only the 195 instances with empty `model_patch`, those entries had to be **deleted** from `preds.json` first:

```python
# Pre-rerun prune (equivalent of what was done):
import json
d = json.load(open('experiments/qwen3_baseline_verified/preds.json'))
# backup first
json.dump(d, open('experiments/qwen3_baseline_verified/preds.json.final_submitted', 'w'), indent=2)
empties = [k for k,v in d.items() if not v.get('model_patch')]  # exactly 195
for k in empties: del d[k]
json.dump(d, open('experiments/qwen3_baseline_verified/preds.json', 'w'), indent=2)
# also clean 10 stale trajectory dirs from those empties so retry doesn't read cache
```

The 195-entry empty list is snapshotted in [to_rerun.txt](to_rerun.txt) (verifiable as `set(to_rerun.txt) == {k for k,v in preds.json if not v['model_patch']}` against `preds.json.final_submitted`).

On rerun, the agent logs `Skipping 305 existing instances` / `Running on 195 instances...` — consistent with the prune.

### Timeline (all UTC)
| When | Event | Outcome |
|---|---|---|
| 2026-04-20 22:28 | Initial run start (4 workers) | |
| 2026-04-21 00:31 | Initial run pass 1 ends (500/500 processed, 2h02m) | **Exit statuses: 293 CalledProcessError, 205 Submitted, 1 TimeoutExpired, 1 ContextWindowExceededError** — docker daemon was unstable under 4-worker contention |
| 2026-04-21 03:28 | Resume pass (same script's while-loop) | Re-runs the 295 `CalledProcessError`/`TimeoutExpired`/`ContextWindowExceededError` instances |
| 2026-04-21 04:45 | Pass 2 ends (295/295, 1h17m) | **Exit statuses: 185 CalledProcessError, 110 Submitted** → aggregate `preds.json`: 305 non-empty / 195 empty |
| 2026-04-21 07:45 | First sb-cli submission as `qwen3_coder_baseline_final` | 83/500 = 16.60% resolved |
| 2026-04-22 05:18 | Rerun for 195 empties (after prune, 3 polite workers on cleaned docker) | |
| 2026-04-22 08:32 | Rerun ends (195/195, 3h14m) | **Exit statuses: 193 Submitted, 2 ContextWindowExceededError** → 188 new non-empty patches added → 493 non-empty / 7 empty |
| 2026-04-22 17:13 | Resubmit as `qwen3_coder_baseline_rerun_complete` | 146/500 = 29.20% resolved (+63 vs prior; 0 lost resolves) |

## Output artifacts

All paths relative to repo root.

| Artifact | Path | Notes |
|---|---|---|
| Per-instance trajectories | `experiments/qwen3_baseline_verified/<instance_id>/<instance_id>.traj.json` | 500 trajectories |
| Agent log | `experiments/qwen3_baseline_verified/minisweagent.log` | 1.2 MB |
| Final preds.json | `experiments/qwen3_baseline_verified/preds.json` | 500 entries, 493 non-empty |
| Pre-rerun backup | `experiments/qwen3_baseline_verified/preds.json.final_submitted` | Same file after first submission |
| Earlier backup | `experiments/qwen3_baseline_verified/preds.json.before_retry` | Mid-run snapshot |
| Initial run log | `experiments/logs/verified.log` | 2 MB |
| Rerun log | `experiments/logs/rerun_empties.log` | 192 KB |
| sb-cli submit logs | `experiments/logs/sb_cli_rerun_submit.log` | |
| vLLM server log | `experiments/logs/vllm_server.log` | |
| sb-cli final report | `sb-cli-reports/Subset.swe_bench_verified__test__qwen3_coder_baseline_rerun_complete.json` | |
| sb-cli prior report | `sb-cli-reports/Subset.swe_bench_verified__test__qwen3_coder_baseline_final.json` | 83/500 |

## Known caveats (for audit)

1. **Two sequential runs, not one.** The 195 empty-patch instances were re-run after initial docker daemon crashes. No previously-resolved instance regressed in the rerun (0 lost resolves). sb-cli deduplicates by instance_id, so the merged preds.json is the canonical submission.

2. **Patch format breakdown** (all 500 instances, counted from final [preds.json](qwen3_baseline_verified/preds.json)):

   | Bucket | Count | Criteria |
   |---|---|---|
   | A. Starts with `diff --git` | 372 | `patch.startswith('diff --git')` |
   | B. Embedded `diff --git` (prose before diff) | 10 | `not A and 'diff --git' in patch` |
   | C. Unified diff without `diff --git` header | 12 | `not A and not B and '--- a/' in patch and '+++ b/' in patch` |
   | D. Not a patch at all (prose / summary / code) | 99 | everything else, non-empty |
   | E. Empty submission | 7 | `patch == ''` |

   Buckets A + B + (parseable subset of C) = patches sb-cli can try to apply. D is 99 cases where Qwen3-Coder-30B wrote things like `=== SUMMARY OF CHANGES ===` or plain English descriptions instead of a git diff. This is a **model behavior** issue with submit-protocol compliance, not a mini-swe-agent bug — the upstream `swebench.yaml` explicitly instructs `git diff ... > patch.txt` followed by `echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT && cat patch.txt`, and mini-swe-agent faithfully captures whatever stdout the model produces after the marker.

   (Independent audit using a slightly looser header definition reported 13 C / 98 D; the 1-instance boundary is immaterial to the 29.20% result.)

3. **7 empty submissions — actually:**

   | Instance | `info.exit_status` | Why empty |
   |---|---|---|
   | matplotlib__matplotlib-26208 | `ContextWindowExceededError` | Hit vLLM 65k context before submitting |
   | sphinx-doc__sphinx-9658 | `ContextWindowExceededError` | Same |
   | pytest-dev__pytest-7324 | `Submitted` | Model ran `echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT` **without** `&& cat patch.txt` — submit marker captured, patch not captured |
   | pytest-dev__pytest-7490 | `Submitted` | Same pattern — submit protocol misuse |
   | sphinx-doc__sphinx-9602 | `Submitted` | Same |
   | sympy__sympy-13878 | `Submitted` | Same |
   | sympy__sympy-15809 | `Submitted` | Same |

   So only 2 are context-window-exceeded; the other 5 are submit-protocol compliance failures (model created a valid `patch.txt` but omitted `cat patch.txt` from the final submit command). None of the 7 hit the step_limit=250.

4. **Code modifications.** `git status` shows the upstream tracked files under `src/minisweagent/` are unchanged (no modifications, no staged diff). There *are* untracked local files in the tree — `src/minisweagent/agents/memory_search.py`, `src/minisweagent/agents/planmem/`, `src/minisweagent/agents/planmem_agent.py` — but these are drafts for a future planning/memory variant and were **not** imported or invoked during this baseline. Evidence: every one of the 500 trajectories' `info.agent_class` / `info.model_class` is the upstream `ProgressTrackingAgent` / `LitellmModel` (verifiable by grepping `experiments/qwen3_baseline_verified/*/[^.]*.traj.json`). The default `swebench.yaml` is used verbatim.

5. **vLLM `--max-model-len 65536`** (not 128k / 256k), chosen to fit 2-GPU tensor-parallel at 0.65 GPU memory utilization. The litellm registry ([registry.json](registry.json)) advertises 262144 tokens but this is **only** `litellm.utils.register_model(...)` metadata (see [src/minisweagent/models/litellm_model.py:60-61](../src/minisweagent/models/litellm_model.py#L60-L61)) used by LiteLLM for routing/cost — mini-swe-agent does **not** use it for prompt trimming. Actual context cap is enforced by vLLM (confirmed: 2 `ContextWindowExceededError` instances hit the 65k limit).

6. **vLLM tool-call parser.** Using `--tool-call-parser qwen3_coder`. Current vLLM docs ([tool calling](https://docs.vllm.ai/en/latest/features/tool_calling/)) also list `qwen3_xml`; the vLLM build in this env accepts `qwen3_coder` and the server loaded without error. If reproduction uses a newer vLLM, double-check this flag against the docs for your version.

7. **temperature=0.0, single attempt.** No sampling, no self-consistency, no best-of-N, no majority vote. Directly comparable to other single-attempt entries on the [swebench.com](https://www.swebench.com) leaderboard.

## Comparison reference (SWE-bench Verified, same model Qwen3-Coder-30B-A3B)

| Scaffold | Attempts | Resolved | Source |
|---|---|---|---|
| **mini-swe-agent v2.2.8 (this run)** | 1 | **29.20%** | — |
| OpenHands | 1 | 51.60% | `20250805_openhands-Qwen3-Coder-30B-A3B-Instruct` |
| R2E-Gym + EntroPO | 1 | 52.20% | `20250901_entroPO_R2E_QwenCoder30BA3B` |
| R2E-Gym + EntroPO + TTS | 2+ | 60.40% | `20250901_entroPO_R2E_QwenCoder30BA3B_tts` |

Data from https://github.com/swe-bench/experiments/tree/main/evaluation/verified

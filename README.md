# mini-swe-agent local changes

For the preserved upstream project README, see [README.upstream.md](README.upstream.md).

This repository is based on the upstream `SWE-agent/mini-swe-agent` project and keeps the original codebase plus local evaluation artifacts.

Changes made to the original repository files:

1. `src/minisweagent/config/benchmarks/swebench.yaml`
   Default benchmark model changed from `anthropic/claude-sonnet-4-5-20250929` to `deepseek/deepseek-chat`.

2. `src/minisweagent/environments/docker.py`
   Docker image pull timeout changed from `120` seconds to `7200` seconds to support slower or larger image pulls during evaluation.

3. `.gitignore`
   Adjusted so the repository keeps evaluation outputs under `output/` and evaluation result files under `logs/run_evaluation/`, while excluding local-only environment/tooling directories such as `.tools/` and `.wsl-eval-venv/`. The `scripts/` directory is kept for reusable evaluation helper scripts, while `scripts/__pycache__/` remains ignored.

Evaluation artifacts included in this repo:

1. `logs/run_evaluation/deepseek_bench_eval/`
2. `logs/run_evaluation/deepseek_bench1_eval/`
3. `logs/run_evaluation/deepseek_bench1_eval/summary/`
   Contains the Markdown summary for the current `deepseek__deepseek-chat` evaluation results.
4. `output/deepseek_bench_dev/`
5. `output/deepseek_bench1/`
6. `output/deepseek_lite_dev/`

Evaluation result summary notes:

1. Summary file for the current run:
   `logs/run_evaluation/deepseek_bench1_eval/summary/deepseek__deepseek-chat_results.md`
2. The summary is derived from per-instance `report.json` files under:
   `logs/run_evaluation/deepseek_bench1_eval/deepseek__deepseek-chat/`
3. Recommended command to re-check the aggregated results:
   `python scripts\check_eval_results.py logs\run_evaluation\deepseek_bench1_eval\deepseek__deepseek-chat`
4. `scripts/check_eval_results.py` reads each instance-level `report.json` and aggregates:
   `resolved`, `patch_successfully_applied`, `FAIL_TO_PASS`, `PASS_TO_PASS`, and `PASS_TO_FAIL`, then prints a compact PASS/FAIL summary table.

Local-only files intentionally not included:

1. `.tools/`
2. `.wsl-eval-venv/`
3. root-level temporary files such as `preds.json`, `traj.json`, `exit_statuses*.yaml`, and `minisweagent.log`

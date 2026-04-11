# mini-swe-agent local changes

This repository is based on the upstream `SWE-agent/mini-swe-agent` project and keeps the original codebase plus local evaluation artifacts.

Changes made to the original repository files:

1. `src/minisweagent/config/benchmarks/swebench.yaml`
   Default benchmark model changed from `anthropic/claude-sonnet-4-5-20250929` to `deepseek/deepseek-chat`.

2. `src/minisweagent/environments/docker.py`
   Docker image pull timeout changed from `120` seconds to `7200` seconds to support slower or larger image pulls during evaluation.

3. `.gitignore`
   Adjusted so the repository keeps evaluation outputs under `output/` and evaluation result files under `logs/run_evaluation/`, while excluding local-only environment/tooling directories such as `.tools/`, `.wsl-eval-venv/`, and temporary scripts.

Evaluation artifacts included in this repo:

1. `logs/run_evaluation/deepseek_bench_eval/`
2. `logs/run_evaluation/deepseek_bench1_eval/`
3. `output/deepseek_bench_dev/`
4. `output/deepseek_bench1/`
5. `output/deepseek_lite_dev/`

Local-only files intentionally not included:

1. `.tools/`
2. `.wsl-eval-venv/`
3. root-level temporary files such as `preds.json`, `traj.json`, `exit_statuses*.yaml`, and `minisweagent.log`

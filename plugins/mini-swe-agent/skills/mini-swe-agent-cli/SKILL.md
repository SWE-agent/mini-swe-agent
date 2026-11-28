---
name: mini-swe-agent-cli
description: >-
  Install and run the mini-SWE-agent `mini` CLI: model and API key setup with
  mini-extra config, confirm/yolo/human modes, choosing a YAML config, cost and
  step limits, and browsing trajectories with the inspector. Use when someone
  wants to run mini-SWE-agent on a local task or tune its global settings.
---

# Running `mini`

References: `docs/quickstart.md`, `docs/usage/mini.md`,
`docs/advanced/global_configuration.md`, `docs/usage/inspector.md`.

## Install

```bash
pip install mini-swe-agent           # or:
uvx mini-swe-agent                   # run without installing
uvx --from mini-swe-agent mini-extra # extra utilities
pipx install mini-swe-agent
```

## First-time setup

```bash
mini-extra config setup                                   # default model + API key
mini-extra config set MSWEA_MODEL_NAME "anthropic/claude-sonnet-4-5-20250929"
mini-extra config set ANTHROPIC_API_KEY "sk-..."
mini-extra config unset ANTHROPIC_API_KEY
mini-extra config edit                                    # open the .env file
```

Global settings live in a `.env` file (path printed when `mini` starts);
exported environment variables win over it. Models go through litellm, plus
OpenRouter and Portkey.

## Run a task

```bash
mini -t "Fix the failing test in tests/test_io.py"
mini -m openai/gpt-5 -c mini.yaml
mini -y -t "..."     # start in yolo mode
```

Flags: `-t/--task`, `-c/--config` (default `mini.yaml` or
`MSWEA_MINI_CONFIG_PATH`), `-m/--model` (default `MSWEA_MODEL_NAME`),
`-y/--yolo`.

`-c` can be repeated to layer overrides on a base config. Always include the
base file first:

```bash
mini -c mini.yaml -c model.model_kwargs.temperature=0.5
```

## Modes

| Mode | Command | Behaviour |
|------|---------|-----------|
| confirm (default) | `/c` | Each proposed command waits for Enter or a rejection message |
| yolo | `/y` | Commands run immediately |
| human | `/u` | You type and run commands yourself |

`Ctrl+C` interrupts the agent so you can switch modes. `mini` runs in the
**local** environment with no sandbox: prefer confirm mode on a real machine.

## Spending guards

- Per run: `agent.cost_limit` (default `3.` in `mini.yaml`) and
  `agent.step_limit` (0 = unlimited) in the YAML config.
- Global: `MSWEA_GLOBAL_COST_LIMIT`, `MSWEA_GLOBAL_CALL_LIMIT`.
- `MSWEA_COST_TRACKING="ignore_errors"` only for free/local models; it
  disables a safety check.

## Inspect a run

`mini` saves the last run's trajectory to the global config directory.

```bash
mini-extra inspector          # find *.traj.json under the current directory
mini-e i path/to/run.traj.json
```

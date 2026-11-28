---
name: mini-swe-agent-python
description: >-
  Use mini-SWE-agent as a Python library and customise it through YAML agent
  configs: DefaultAgent, get_model, environment classes, Jinja2 prompt
  templates and the agent/environment/model/run config sections. Use when
  embedding mini-SWE-agent in a pipeline, writing a custom config, or
  subclassing an agent.
---

# mini-SWE-agent as a library

References: `docs/usage/python_bindings.md`, `docs/advanced/yaml_configuration.md`,
`docs/advanced/cookbook.md`, `docs/advanced/environments.md`,
`src/minisweagent/config/*.yaml`.

## Hello world

```python
from minisweagent.agents.default import DefaultAgent
from minisweagent.models import get_model
from minisweagent.environments.local import LocalEnvironment

agent = DefaultAgent(
    get_model(input_model_name="anthropic/claude-sonnet-4-5-20250929"),
    LocalEnvironment(),
)
agent.run("Write a hello world program")
```

The agent only runs bash commands; there are no custom tools to register.
That is a design choice: keep extensions in the environment or prompts.

## Environments

| Class key | Use |
|-----------|-----|
| `local` | `subprocess.run` on the host, no isolation |
| `docker` | `docker exec` in a container |
| `singularity` | Apptainer/Singularity on HPC |
| `swerex_docker`, `swerex_modal` | via SWE-ReX; Modal for scale |
| `bubblewrap` | Linux-only lightweight sandbox (experimental) |
| `contree` | ConTree sandbox |

Choose with `environment.environment_class` in YAML or
`--environment-class` on run scripts.

## YAML config structure

Top-level keys:

- `agent`: templates, `step_limit`, `cost_limit`, `mode`, optional
  `agent_class` (name or import path for your own class).
- `environment`: `environment_class`, `cwd`, `timeout`, `env` variables.
- `model`: model name/class, `model_kwargs`, `observation_template`.
- `run`: run-script options (output file, `env_startup_command`).

Start from `src/minisweagent/config/mini.yaml` (interactive) or
`default.yaml`, copy it, and pass it with `-c`.

## Prompt templates

Templates are Jinja2 with double braces, e.g. `{{task}}`. Available
variables: agent and environment config values, variables passed to
`run()` (by default `task`), the last action's `output`
(`output.returncode`, `output.output`), and host environment variables
for `LocalEnvironment`. Truncate long observations in the template, as the
shipped `observation_template` does (head/tail with an elided-chars count).

## Subclassing

Prefer composition: swap the model, environment or templates first. For
control-flow changes, subclass `DefaultAgent` and point `agent.agent_class`
at your import path. See `docs/advanced/cookbook.md` and
`docs/advanced/control_flow.md`.

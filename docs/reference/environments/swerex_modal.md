# SWE-ReX Modal

!!! note "SWE-ReX Modal Environment class"

    - [Read on GitHub](https://github.com/swe-agent/mini-swe-agent/blob/main/src/minisweagent/environments/extra/swerex_modal.py)
    - Requires [Modal](https://modal.com) account and authentication

This environment executes commands in [Modal](https://modal.com) sandboxes using [SWE-ReX](https://github.com/swe-agent/swe-rex).

## Setup

1. Install the full dependencies:
   ```bash
   pip install "mini-swe-agent[full]"
   pip install modal
   ```

2. Set up Modal authentication:
   ```bash
   modal setup
   ```

## Usage

You can use this environment class with the `--environment-class` flag:

```bash
mini --environment-class swerex_modal --environment.image python:3.11-slim
```

Or in your agent config file:

```yaml
environment:
  environment_class: swerex_modal
  image: python:3.11-slim
  timeout: 60
  runtime_timeout: 3600
```

## Use Cases

The Modal environment is particularly useful for:

- **Training coding agents**: Run agents at scale with isolated cloud environments
- **Parallel evaluation**: Run many agent instances in parallel on SWE-bench
- **CI/CD integration**: Execute tests and evaluations in clean, reproducible environments

::: minisweagent.environments.extra.swerex_modal

{% include-markdown "../../_footer.md" %}

# E2B

!!! note "E2B Environment class"

    - [Read on GitHub](https://github.com/swe-agent/mini-swe-agent/blob/main/src/minisweagent/environments/extra/e2b.py)
    - Requires an [E2B](https://e2b.dev) account and API key

    ??? note "Full source code"

        ```python
        --8<-- "src/minisweagent/environments/extra/e2b.py"
        ```

::: minisweagent.environments.extra.e2b

This environment executes commands in [E2B](https://e2b.dev) cloud sandboxes.
E2B converts Docker images into persistent sandbox templates, so **no local Docker daemon is required** — everything runs in the cloud.

This makes it well-suited for:

- Large-scale, fully-remote SWE-bench evaluations
- Environments where Docker is unavailable (CI, serverless)
- Parallel agent runs without managing local container infrastructure

## How it works

The first time a Docker image is used, `E2BEnvironment` builds a persistent E2B template from that image (via `Template.build`). Subsequent runs reuse the cached template, so the build cost is paid only once per unique image.

## Setup

1. Install the E2B extra:
   ```bash
   pip install "mini-swe-agent[e2b]"
   ```

2. Set your E2B API key:
   ```bash
   export E2B_API_KEY="your-e2b-api-key"
   ```

## Usage

Evaluate on SWE-bench using E2B as the sandbox backend:
```bash
mini-extra swebench \
    --subset verified \
    --split test \
    --workers 50 \
    --environment-class e2b
```

Or specify it in your YAML config:
```yaml
environment:
  environment_class: e2b
  sandbox_timeout: 3600  # seconds the sandbox stays alive
  cpu_count: 2
  memory_mb: 2048
```

## Configuration reference

| Field | Default | Description |
|-------|---------|-------------|
| `image` | *(required)* | Docker Hub image to use as the sandbox base |
| `cwd` | `/` | Default working directory for commands |
| `timeout` | `30` | Per-command timeout in seconds |
| `env` | `{}` | Environment variables set in every command |
| `sandbox_timeout` | `3600` | How long the sandbox stays alive (seconds) |
| `cpu_count` | `2` | vCPUs allocated to the sandbox |
| `memory_mb` | `2048` | Memory allocated to the sandbox (MiB) |
| `build_timeout` | `1800` | Max seconds to wait for a template build |
| `skip_cache` | `False` | Force-rebuild the template even if it exists |
| `api_key` | `None` | E2B API key (falls back to `E2B_API_KEY` env var) |
| `registry_username` | `None` | Username for private Docker registry auth |
| `registry_password` | `None` | Password for private Docker registry auth |

{% include-markdown "../../_footer.md" %}

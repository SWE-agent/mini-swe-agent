# Runtime

!!! note "Runtime Environment class"

    - [Read on GitHub](https://github.com/swe-agent/mini-swe-agent/blob/main/src/minisweagent/environments/extra/runtime.py)
    - Requires a [Runtime](https://withruntime.com) API key

    ??? note "Full source code"

        ```python
        --8<-- "src/minisweagent/environments/extra/runtime.py"
        ```

::: minisweagent.environments.extra.runtime

This environment executes commands in [Runtime](https://withruntime.com) sandboxes, Firecracker microVMs with their own Linux kernel, using the [withruntime](https://pypi.org/project/withruntime/) SDK.

## Setup

1. Install the dependencies:

   ```bash
   pip install "mini-swe-agent[runtime]"
   ```

2. Set your API key, or run `npx withruntime login` once on the machine:
   ```bash
   export RUNTIME_API_KEY="your-runtime-api-key"
   ```

## Usage

Run mini-swe-agent like with any other environment:

```
mini-extra swebench \
    --subset verified \
    --split test \
    --workers 8 \
    --environment-class runtime
```

Each instance's SWE-bench image is built into a Runtime image the first time it is used and found again after that, so a rerun starts at once. Commands run as `root` through the sandbox user's passwordless sudo; set `user: null` to run them as the sandbox user.

Other options go under `environment` in your config: `vcpu`, `memory_mib`, `runtime_image` (start from an existing Runtime image) and `create` for any other sandbox setting, such as `{"disk_mib": 16384}`. Each sandbox keeps a lease ten minutes ahead while the run is alive and stops when the environment is cleaned up, so a crashed run leaves nothing running.

{% include-markdown "../../_footer.md" %}

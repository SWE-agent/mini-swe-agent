# `micro2`

`micro2` is a pager-style interactive command line interface for using micro-SWE-agent in the local requirement (as opposed for workflows that require sandboxing or large scale batch processing).
Compared to [`micro`](micro.md), `micro2` offers a more advanced UI based on [Textual](https://textual.textualize.io/).

VIDEO HERE TODO

!!! tip "Feedback wanted!"
    Give feedback on the `micro` and `micro2` interfaces at [this github issue](https://github.com/swe-agent/micro-swe-agent/issues/161)
    or in our [Slack channel](https://join.slack.com/t/swe-bench/shared_invite/zt-36pj9bu5s-o3_yXPZbaH2wVnxnss1EkQ).

## Command line options

Useful switches:

- `-h`/`--help`: Show help
- `-t`/`--task`: Specify a task to run (else you will be prompted)
- `-c`/`--config`: Specify a config file to use, else we will use [`local2.yaml`](https://github.com/swe-agent/micro-swe-agent/blob/main/src/microsweagent/config/local2.yaml) or the config `MSWEA_LOCAL2_CONFIG_PATH` environment variable (see [configuration](../advanced/configuration.md))
  It's enough to specify the name of the config file, e.g., `-c local.yaml` (see [configuration](../advanced/configuration.md) for how it is resolved).
- `-m`/`--model`: Specify a model to use, else we will use the model `MSWEA_MODEL_NAME` environment variable (see [configuration](../advanced/configuration.md))
- `-y`/`--yolo`: Start in `yolo` mode (see below)

## Key bindings

- `q`: Quit the agent
- `c`: Switch to `confirm` mode
- `y`: Switch to `yolo` mode
- `h` or `LEFT`: Go back in history
- `l` or `RIGHT`: Go forward in history
- `j` or `DOWN`: Scroll down
- `k` or `UP`: Scroll up

## Modes of operation

`micro2` provides two different modes of operation

- `confirm` (`c`): The LM proposes an action and the user is prompted to confirm (press Enter)) or reject (enter a rejection message))
- `yolo` (`y`): The action from the LM is executed immediately without confirmation

You can switch between the modes at any time by pressing the `c` or `y` keys.

`micro2` starts in `confirm` mode by default. To start in `yolo` mode, you can add `-y`/`--yolo` to the command line.

## FAQ

> How can I copy output from the

## Implementation

??? note "Default config"

    - [Read on GitHub](https://github.com/swe-agent/micro-swe-agent/blob/main/src/microsweagent/config/local.yaml)

    ```yaml
    --8<-- "src/microsweagent/config/local.yaml"
    ```

??? note "Run script"

    - [Read on GitHub](https://github.com/swe-agent/micro-swe-agent/blob/main/src/microsweagent/run/micro.py)
    - [API reference](../reference/run/micro.md)

    ```python
    --8<-- "src/microsweagent/run/micro.py"
    ```

??? note "Agent class"

    - [Read on GitHub](https://github.com/swe-agent/micro-swe-agent/blob/main/src/microsweagent/agents/interactive.py)
    - [API reference](../reference/agents/interactive.md)

    ```python
    --8<-- "src/microsweagent/agents/interactive.py"
    ```

{% include-markdown "../_footer.md" %}

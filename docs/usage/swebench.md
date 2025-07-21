# SWE-bench

!!! abstract "Overview"

    * We provide two scripts to run on the [SWE-bench](https://www.swebench.com/) benchmark.
    * `micro-extra swebench` runs on all task instances in batch mode.
    * `micro-extra swebench-single` runs on a single task instance with interactivity (useful for debugging).

## Usage

=== "Batch mode"

    ```bash
    micro-extra swebench
    # or
    python src/microsweagent/run/extra/swebench.py
    # show help
    micro-extra swebench --help
    # for example
    micro-extra swebench \
        --model claude-sonnet-4-20250514 \
        --subset verified \
        --split test \
        --workers 4
    ```

=== "Single instance (for debugging)"

    ```bash
    micro-extra swebench-single
    # or
    python src/microsweagent/run/extra/swebench_single.py
    # show help
    micro-extra swebench-single --help
    # for example
    micro-extra swebench-single \
        --model claude-sonnet-4-20250514 \
        --subset verified \
        --split test \
        -i 0  # instance index (alternatively, specify instance name)
    ```

## FAQ

> What happens to uncompleted tasks when I abort with KeyboardInterrupt?

Trajectories are only saved upon completion, so most likely, you can just rerun the script to complete the tasks next time.
However, you should still check for `KeyboardInterrupt` in `preds.json` in case some tasks were aborted but saved.

> Certain tasks are being stuck even though I deleted the trajectories.

The completed instances are inferred from `preds.json`. Remove the corresponding items from the file.

> How can I run on a different dataset?

As long as it follows the SWE-bench format, you can use `--subset /path/to/your/dataset` to run on a custom dataset.
The dataset needs to be loadable as `datasets.load_dataset(path, split=split)`.

> Some progress runners are stuck at 'initializing task' for a very long time

They might be pulling docker containers -- the run sshould start immediately the next time.

## Implementation

??? note "Default config"

    - [Read on GitHub](https://github.com/swe-agent/micro-swe-agent/blob/main/src/microsweagent/config/extra/swebench.yaml)

    ```yaml
    --8<-- "src/microsweagent/config/extra/swebench.yaml"
    ```

??? note "`swebench.py` run script"

    - [Read on GitHub](https://github.com/swe-agent/micro-swe-agent/blob/main/src/microsweagent/run/extra/swebench.py)
    - [API reference](../reference/run/swebench.md)

    ```python
    --8<-- "src/microsweagent/run/extra/swebench.py"
    ```

??? note "`swebench_single.py` run script"

    - [Read on GitHub](https://github.com/swe-agent/micro-swe-agent/blob/main/src/microsweagent/run/extra/swebench_single.py)
    - [API reference](../reference/run/swebench_single.md)

    ```python
    --8<-- "src/microsweagent/run/extra/swebench_single.py"
    ```

{% include-markdown "../_footer.md" %}
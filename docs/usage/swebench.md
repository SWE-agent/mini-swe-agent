# SWE-bench

We provide two scripts to run on [SWE-bench](https://www.swebench.com/):

- `src/microsweagent/run/extra/swebench.py`: Evaluate on SWE-bench in batch mode
- `src/microsweagent/run/extra/swebench_single.py`: Evaluate on a single SWE-bench task
   (used for development & debugging).

## FAQ

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
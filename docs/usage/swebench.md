# SWE-bench

!!! abstract "Overview"

    * We provide two scripts to run on the [SWE-bench](https://www.swebench.com/) benchmark.
    * `mini-extra swebench` runs on all task instances in batch mode.
    * `mini-extra swebench-single` runs on a single task instance with interactivity (useful for debugging).

<figure markdown="span">
  <div class="gif-container gif-container-styled" data-glightbox-disabled>
    <img src="https://github.com/SWE-agent/swe-agent-media/blob/main/media/mini/png/swebench.png?raw=true"
         data-gif="https://github.com/SWE-agent/swe-agent-media/blob/main/media/mini/gif/swebench.gif?raw=true"
         alt="swebench" data-glightbox="false" width="600" />
  </div>
</figure>

## Usage

=== "Batch mode"

    ```bash
    mini-extra swebench --help
    # or
    python src/minisweagent/run/extra/swebench.py --help
    # Example:
    mini-extra swebench \
        --model claude-sonnet-4-20250514 \
        --subset verified \
        --split test \
        --workers 4
    ```

=== "Single instance (for debugging)"

    ```bash
    mini-extra swebench-single --help
    # or
    python src/minisweagent/run/extra/swebench_single.py --help
    # Example:
    mini-extra swebench-single \
        --subset verified \
        --split test \
        --model claude-sonnet-4-20250514 \
        -i sympy__sympy-15599
    # or
    mini-extra swebench-single \
        --subset verified \
        --split test \
        -m claude-sonnet-4-20250514 \
        -i 0  # instance index
    ```

!!! tip "Evaluating on SWE-bench"

    You can use the [sb-cli](https://www.swebench.com/sb-cli/) for extremely fast, cloud-based evaluations
    (and it's free!). After installing it and getting a token, simply run:

    ```bash
    sb-cli submit swe-bench_verified test --predictions_path preds.json --run_id some-id-for-your-run
    ```

    Typically you will have results within 20 minutes (this is not limited by how many instances you run,
    but by the slowest-to-evaluate instance in SWE-bench).


## FAQ

> Can I set global cost limits?

Yes, you can set global cost limits with the `MSWEA_GLOBAL_CALL_LIMIT` and `MSWEA_GLOBAL_COST_LIMIT` environment variables/global config.
See [configuration](../advanced/configuration.md) for more details.

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

    - [Read on GitHub](https://github.com/swe-agent/mini-swe-agent/blob/main/src/minisweagent/config/extra/swebench.yaml)

    ```yaml
    --8<-- "src/minisweagent/config/extra/swebench.yaml"
    ```

??? note "`swebench.py` run script"

    - [Read on GitHub](https://github.com/swe-agent/mini-swe-agent/blob/main/src/minisweagent/run/extra/swebench.py)
    - [API reference](../reference/run/swebench.md)

    ```python
    --8<-- "src/minisweagent/run/extra/swebench.py"
    ```

??? note "`swebench_single.py` run script"

    - [Read on GitHub](https://github.com/swe-agent/mini-swe-agent/blob/main/src/minisweagent/run/extra/swebench_single.py)
    - [API reference](../reference/run/swebench_single.md)

    ```python
    --8<-- "src/minisweagent/run/extra/swebench_single.py"
    ```

{% include-markdown "../_footer.md" %}
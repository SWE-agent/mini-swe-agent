# SWE-bench

!!! abstract "Overview"

    * We provide two scripts to run on the [SWE-bench](https://www.swebench.com/) benchmark.
    * `mini-extra swebench` runs on all task instances in batch mode.
    * `mini-extra swebench-single` runs on a single task instance with interactivity (useful for debugging).
    * You can also take a look at the runscripts to figure out how to build your own batch processing pipeline.

<figure markdown="span">
  <div class="gif-container gif-container-styled" data-glightbox-disabled>
    <img src="https://github.com/SWE-agent/swe-agent-media/blob/main/media/mini/png/swebench.png?raw=true"
         data-gif="https://github.com/SWE-agent/swe-agent-media/blob/main/media/mini/gif/swebench.gif?raw=true"
         alt="swebench" data-glightbox="false" width="600" />
  </div>
</figure>

## Usage

!!! warning "Docker container availability"

    The docker containers for Linux assume an x86 Linux architecture;
    you might not be able to run them on other architectures.


!!! tip "Quickstart"

    === "Batch mode"

        Batch mode runs on all task instances in parallel.

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

        Basic flags:

        - `-o`, `--output` - Output directory
        - `-m`, `--model` - Model to use
        - `-c`, `--config` - Path to a config file (default: `swebench.yaml` in the `config` directory)
        - `-w`, `--workers` - Number of worker threads for parallel processing (default: `1`)

        Data selection flags:

        - `--subset` - SWEBench subset to use or path to a dataset (default: `lite`)
        - `--split` - Dataset split (default: `dev`)
        - `--slice` - Slice specification (e.g., '0:5' for first 5 instances)
        - `--filter` - Filter instance IDs by regex
        - `--shuffle` - Shuffle instances (default: `False`)
        - `--redo-existing` - Redo existing instances (default: `False`)

        Advanced flags:

        - `--environment-class` - Environment type to use (recommended: `docker` or `singularity`)

    === "Single instance (for debugging)"

        Single instance mode runs on a single task instance with interactivity (useful for debugging).

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

        Note: If you want to run the script without prompting for confirmation at exit,
        add the `--exit-immediately` flag.

        Basic flags:

        - `-m`, `--model` - Model to use
        - `-c`, `--config` - Path to a config file (default: `swebench.yaml` in the `config` directory)
        - `-o`, `--output` - Output trajectory file (default: saves to global config directory)

        Data selection flags:

        - `--subset` - SWEBench subset to use or path to a dataset (default: `lite`)
        - `--split` - Dataset split (default: `dev`)
        - `-i`, `--instance` - SWE-Bench instance ID (default: `0`)

        Advanced flags:

        - `--environment-class` - Environment type to use (recommended: `docker` or `singularity`)
        - `--exit-immediately` - Exit immediately when the agent wants to finish instead of prompting (default: `False`)

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

> I have some docker issues

Try running the docker command manually to see what's going on (it should be printed out in the console).
Confirm that it's running with `docker ps`, and that you can use `docker exec -it <container-id> ls` to get some output.

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
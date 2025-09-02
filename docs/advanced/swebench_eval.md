# SWE-bench Evaluation

!!! abstract "Overview"

    * Mini-SWE-Agent provides a native script for evaluating trajectories on SWE-bench.
    * The evaluation script is designed to be minimalistic, and meant to be used in a batch mode.

## Usage

!!! tip "Quickstart"

        ```bash
        mini-extra swebench-eval --help
        # or
        python src/minisweagent/run/extra/swebench_eval.py --help
        # Example:
        mini-extra swebench-eval \
            --model claude-sonnet-4-20250514 \
            --split test \
            --workers 4
            --dataset SumanthRH/SWE-Bench_Verified
        ```

        Basic flags:

        - `-o`, `--output` - Output directory containing generated trajectories in a `preds.json` file. This is the output from `mini-extra swebench` or `mini-extra swebench-single`.
        - `-m`, `--model` - Model to use
        - `-c`, `--config` - Path to a config file (default: `swebench_eval.yaml` in the `config` directory)
        - `-w`, `--workers` - Number of worker threads for parallel processing (default: `1`)

        Data selection flags:

        - `--dataset` - SWEBench dataset to use or path to a dataset. In addition to the standard SWEBench dataset columns, this should contain a `eval_script` column that contains the evaluation script for that instance. (default: `SumanthRH/SWE-Bench_Verified`)
        - `--split` - Dataset split (default: `dev`)
        - `--slice` - Slice specification (e.g., '0:5' for first 5 instances)
        - `--filter` - Filter instance IDs by regex
        - `--shuffle` - Shuffle instances (default: `False`)
        - `--redo-existing` - Redo existing instances (default: `False`)

        Advanced flags:

        - `--environment-class` - Environment type to use (recommended: `docker` or `singularity`)


## Design

`swebench-eval` performs the following for each instance:

* Load the trajectory from the `preds.json` file.
* Initialize the environment for the given instance using the given backend.
* Apply the model's git patch to the working directory in the environment.
* Run the evaluation script for the instance. If the script runs successfully, the instance is considered to be resolved, and unresolved otherwise.

!!! tip "Preparing the evaluation script"

    `swebench-eval` only checks to see if the final return code on running the evaluation script is 0. Ideally, your evaluation script should be written so that this final return code indicates the success or failure of the evaluation. See [SumanthRH/SWE-Bench_Verified](https://huggingface.co/datasets/SumanthRH/SWE-bench_Verified) for reference.

!!! tip "Size of the git patch"

    `swebench-eval` applies the model's git patch by providing it as an in-line argument to the `git apply` command. This means that the size of the git patch is limited to the OS limits for `ARG_MAX`.
    In modern systems, this is typically ~ 1 MB, which is pretty generous.
    For simplicity, we assume that large patches greater than `ARG_MAX` are meant to fail.

## Implementation

??? note "Default config"

    - [Read on GitHub](https://github.com/swe-agent/mini-swe-agent/blob/main/src/minisweagent/config/extra/swebench_eval.yaml)

    ```yaml
    --8<-- "src/minisweagent/config/extra/swebench_eval.yaml"
    ```

??? note "`swebench_eval.py` run script"

    - [Read on GitHub](https://github.com/swe-agent/mini-swe-agent/blob/main/src/minisweagent/run/extra/swebench_eval.py)
    - [API reference](../reference/run/swebench_eval.md)

    ```python
    --8<-- "src/minisweagent/run/extra/swebench_eval.py"
    ```

{% include-markdown "../_footer.md" %}
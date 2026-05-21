# ProgramBench

!!! abstract "Overview"

    * `mini-extra programbench` runs the agent on all [ProgramBench](https://github.com/facebookresearch/programbench) task instances in batch mode.
    * Output is directly compatible with `programbench eval`.
    * ProgramBench is a reverse-engineering benchmark: the agent is dropped into a container
      with a compiled binary and must produce a fresh source codebase that reproduces the
      binary's behavior. Solutions are scored by running tests against the rebuilt executable.

## Usage

!!! warning "Docker container availability"

    The ProgramBench docker containers (published as `programbench/<instance>:task_cleanroom`)
    assume an x86 Linux architecture; you might not be able to run them on other architectures.


!!! warning "Install ``programbench`` first"

    The runner imports ``programbench`` to discover task instances. Install it with
    ``pip install programbench`` (or ``uvx programbench``) before running.


!!! tip "Quickstart"

    ```bash
    mini-extra programbench --help
    # or
    python src/minisweagent/run/benchmarks/programbench.py --help
    # Example:
    mini-extra programbench \
        --model anthropic/claude-sonnet-4-5-20250929 \
        --workers 4
    ```

    Basic flags:

    - `-o`, `--output` - Output directory (default: timestamped `programbench_results_<ts>/`)
    - `-m`, `--model` - Model to use
    - `-c`, `--config` - Path to a config file (default: `programbench.yaml` in the `config` directory)
    - `-w`, `--workers` - Number of worker threads for parallel processing (default: `1`)

    Data selection flags:

    - `--slice` - Slice specification (e.g., `0:5` for first 5 instances)
    - `--filter` - Filter instance IDs by regex
    - `--shuffle` - Shuffle instances (default: `False`)
    - `--redo-existing` - Redo existing instances (default: `False`)

    Advanced flags:

    - `--environment-class` - Environment type to use (recommended: `docker` or `singularity`)
    - `--model-class` - Model class to use

    The docker image tag is hardcoded to ``:task_cleanroom`` (a build-artifact-free image).

!!! tip "Output layout"

    Each instance writes two files under ``<output>/<instance_id>/``:

    - ``submission.tar.gz`` - the agent's workspace (gzipped tar of ``/workspace`` inside the container)
    - ``<instance_id>.traj.json`` - the full agent trajectory

    The directory can be passed directly to ``programbench eval``:

    ```bash
    programbench eval <output>/
    ```

!!! tip "Network isolation"

    The default config launches each container with ``--network none``. The agent therefore
    cannot install dependencies from the internet, clone GitHub repos, or download source
    tarballs - the entire reverse-engineering exercise has to happen offline against the
    provided binary and its bundled documentation. If you need to allow specific hosts, override
    ``environment.run_args`` in your config.

## FAQ

> Can I set global cost limits?

Yes, you can set global cost limits with the `MSWEA_GLOBAL_CALL_LIMIT` and `MSWEA_GLOBAL_COST_LIMIT` environment variables/global config.
See [global configuration](../advanced/global_configuration.md) for more details.

> What happens to uncompleted tasks when I abort with KeyboardInterrupt?

Trajectories are saved after every agent step (``agent.output_path`` is set automatically), and
``submission.tar.gz`` is written in the ``finally`` block of each instance. You can just rerun
the script — instances that already have a ``submission.tar.gz`` are skipped unless you pass
``--redo-existing``.

> Certain tasks are stuck even though I deleted the trajectories.

The skip check looks at ``submission.tar.gz``, not ``.traj.json``. Delete the tarball (or the
whole ``<instance_id>/`` directory) to force a rerun.

> Some progress runners are stuck at 'initializing task' for a very long time / time out

They might be pulling docker containers — the run should start immediately the next time.
If you see timeouts because of `docker pull` operations, you might want to increase `environment.pull_timeout`
from the default of `120` (seconds).

> I have some docker issues

Try running the docker command manually to see what's going on (it should be printed out in the console).
Confirm that it's running with `docker ps`, and that you can use `docker exec -it <container-id> ls` to get some output.

> Docker isn't available on my HPC cluster.

You can use the singularity/apptainer backend by setting `environment.environment_class` to `singularity`
in your [agent config file](../advanced/yaml_configuration.md)
or specify `--environment-class singularity` from the command line.

## Implementation

??? note "Default config"

    - [Read on GitHub](https://github.com/swe-agent/mini-swe-agent/blob/main/src/minisweagent/config/benchmarks/programbench.yaml)

    ```yaml
    --8<-- "src/minisweagent/config/benchmarks/programbench.yaml"
    ```

??? note "`programbench.py` run script"

    - [Read on GitHub](https://github.com/swe-agent/mini-swe-agent/blob/main/src/minisweagent/run/benchmarks/programbench.py)
    - [API reference](../reference/run/programbench.md)

    ```python
    --8<-- "src/minisweagent/run/benchmarks/programbench.py"
    ```

{% include-markdown "../_footer.md" %}

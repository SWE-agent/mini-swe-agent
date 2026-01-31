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

    We provide two different scripts: `swebench` and `swebench-single`:

    === "Batch mode"

        Batch mode runs on all task instances in parallel.

        ```bash
        mini-extra swebench --help
        # or
        python src/minisweagent/run/extra/swebench.py --help
        # Example:
        mini-extra swebench \
            --model anthropic/claude-sonnet-4-5-20250929 \
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

        Single instance mode runs on a single task instance with interactivity. This is meant for debugging, and so unlike the batch mode command above, this will not produce a preds.json file.

        ```bash
        mini-extra swebench-single --help
        # or
        python src/minisweagent/run/extra/swebench_single.py --help
        # Example:
        mini-extra swebench-single \
            --subset verified \
            --split test \
            --model anthropic/claude-sonnet-4-5-20250929 \
            -i sympy__sympy-15599
        # or
        mini-extra swebench-single \
            --subset verified \
            --split test \
            -m anthropic/claude-sonnet-4-5-20250929 \
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

    You have two options to evaluate on SWE-bench: Our free cloud-based evaluation or the SWE-bench CLI.

    === "Cloud-based evaluation"

        You can use the [sb-cli](https://www.swebench.com/sb-cli/) for extremely fast, cloud-based evaluations
        (and it's free!). After installing it and getting a token, simply run:

        ```bash
        sb-cli submit swe-bench_verified test --predictions_path preds.json --run_id some-id-for-your-run
        ```

        Typically you will have results within 20 minutes (this is not limited by how many instances you run,
        but by the slowest-to-evaluate instance in SWE-bench).

    === "Local evaluation"

        You can also use a local installation of [SWE-bench](https://github.com/SWE-bench/SWE-bench)
        for evaluation:

        ```bash
        python -m swebench.harness.run_evaluation \
            --dataset_name princeton-nlp/SWE-bench_Verified \
            --predictions_path preds.jsonl \
            --max_workers <num_workers> \
            --run_id <run_id>
        ```

## CLI‑Native Agents (Codex, Claude Code, Gemini)

Use the provided config `src/minisweagent/config/extra/swebench_codex_cli.yaml` to run SWE-bench with a CLI-native async agent. This reuses the mini‑SWE‑agent runners and environments while delegating the code‑editing loop to an external CLI (Codex here) in their yolo mode. The same pattern generalizes to other CLI tools (Claude Code, Gemini) with small tweaks.

Note: CLI‑native agents include their own internal scaffolds/orchestration. This integration does not modify those systems; it standardizes only the surrounding harness (dataset loading, containerized environment, and submission outputs) so open-source and academic community can evaluate different CLIs under the same conditions as mini‑swe‑agent.

### What this config does
- Ensures the CLI binary is available via either host mounts (fast path) or per‑instance install at startup.
- Two-step deterministic process: (1) run the CLI-native agent to solve the task; (2) submit the patch.
- Prepares a prompt for the CLI inside the container (`/tmp/prompt.md`).
- Produces `preds.json` for evaluation with `sb-cli` or the SWE-bench harness.

### Requirements
- API key set via `mini-extra config set` or exported in the following way:
  - Codex: `export OPENAI_API_KEY=...`
  - Claude Code: `export ANTHROPIC_API_KEY=...`
  - Gemini CLI: `export GEMINI_API_KEY=...`
- Optional host mounts for faster startup (already encoded in the config): host `node` and `@openai/codex` modules. If not present, the startup script installs them inside the container.

### Example to Run Codex CLI on SWE-bench

Batch mode:

```bash
mini-extra swebench \
  --config src/minisweagent/config/extra/swebench_codex_cli.yaml \
  --subset verified \
  --split test \
  --workers 4 \
  -o outputs/codex_verified_test
```

Single instance:

```bash
mini-extra swebench-single \
  --config src/minisweagent/config/extra/swebench_codex_cli.yaml \
  --subset verified \
  --split test \
  -i sympy__sympy-15599 \
  -o outputs/codex_single.traj.json
```

Notes:
- The `model` is `deterministic` and encodes two fixed actions: run Codex, then submit the patch.
- Inside the container, the agent runs roughly:
  - `codex exec --model gpt-5 --config model_reasoning_effort=medium -C /testbed --yolo < /tmp/prompt.md`
  - Then: `echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT && git add -A && git diff --cached`
- Adjust the Codex CLI flags/model for your setup.

### Generalizing to Claude Code or Gemini CLI
Replicate the pattern by editing a copy of the config in three places:

1) `environment.forward_env` — forward the appropriate API key into the container.
   - Claude Code: add `ANTHROPIC_API_KEY`
   - Gemini: add `GOOGLE_API_KEY` (or whatever your CLI expects)

2) `run.env_startup_command` — ensure the CLI exists in the container.
   - Option A: mount a host‑installed CLI via `environment.run_args`.
   - Option B: install the CLI at startup (curl/apt/pip/npm) and write a small wrapper in `/usr/local/bin/...`.
   - Keep the prompt creation to `/tmp/prompt.md`, tailored to your CLI’s input format.

3) `model.outputs` — replace the Codex call with your CLI’s non‑interactive invocation against `/testbed`, keep the same submit line.

Example skeleton (pseudocode; adapt to your CLI):

```yaml
environment:
  forward_env:
    - ANTHROPIC_API_KEY   # or GOOGLE_API_KEY
  # run_args:            # optional: mount a host CLI binary
  #   - "-v"; "/path/to/cli:/opt/cli:ro"

run:
  env_startup_command: |
    # Install or expose your CLI here
    cat <<'PROMPT_EOF' > /tmp/prompt.md
    ... your prompt/body ...
    PROMPT_EOF

model:
  model_name: deterministic
  model_class: deterministic
  outputs:
    - |
      THOUGHT: Run my CLI against /testbed.
      ```bash
      my-cli run --non-interactive -C /testbed < /tmp/prompt.md
      ```
    - |
      THOUGHT: Submit final patch.
      ```bash
      echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT && git add -A && git diff --cached
      ```
```

This gives you a simple way to compare CLI‑native agents (Codex, Gemini, Claude Code, etc.) under the same harness conditions (dataset, container, submission) and evaluation flow used by mini‑swe‑agent.

## FAQ

> Can I set global cost limits?

Yes, you can set global cost limits with the `MSWEA_GLOBAL_CALL_LIMIT` and `MSWEA_GLOBAL_COST_LIMIT` environment variables/global config.
See [global configuration](../advanced/global_configuration.md) for more details.

> What happens to uncompleted tasks when I abort with KeyboardInterrupt?

Trajectories are only saved upon completion, so most likely, you can just rerun the script to complete the tasks next time.
However, you should still check for `KeyboardInterrupt` in `preds.json` in case some tasks were aborted but saved.

> Certain tasks are being stuck even though I deleted the trajectories.

The completed instances are inferred from `preds.json`. Remove the corresponding items from the file.

> How can I run on a different dataset?

As long as it follows the SWE-bench format, you can use `--subset /path/to/your/dataset` to run on a custom dataset.
The dataset needs to be loadable as `datasets.load_dataset(path, split=split)`.

> Some progress runners are stuck at 'initializing task' for a very long time / time out

They might be pulling docker containers -- the run should start immediately the next time.
If you see timeouts because of `docker pull` operations, you might want to increase `environment.pull_timeout`
from the default of `120` (seconds).

> I have some docker issues

Try running the docker command manually to see what's going on (it should be printed out in the console).
Confirm that it's running with `docker ps`, and that you can use `docker exec -it <container-id> ls` to get some output.

> Docker isn't available on my HPC cluster.

You can use the singularity/apptainer backend by setting `environment.environment_class` to `singularity`
in your [agent config file](../advanced/yaml_configuration.md)
or specify `--environment-class singularity` from the command line

> Can I run a startup command in the environment?

Yes, you can use the `run.env_startup_command` config option to run a command in the environment before the agent starts.
For example:

```yaml
run:
  env_startup_command: "apt-get update && apt-get install -y python3-pip"
```

The command is rendered with the instance variables as template variables using `jinja2`.
For example, you could use

```yaml
run:
  env_startup_command: "git clone {{ repo_url }} . --force"
```

which might be particularly useful when running with environments like [`bubblewrap`](../reference/environments/bubblewrap.md).

> What environment can I use for SWE-bench?

See [this guide](../advanced/environments.md) for more details.

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

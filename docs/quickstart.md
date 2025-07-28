# Quick start

!!! tip "Installation Options"

    === "pip"

        Use pip to install `mini` in your current environment:

        ```bash
        pip install mini-swe-agent
        ```

        And try our command line interface

        ```bash
        mini  # simple UI
        mini -v  # visual UI
        mini-extra  # extra utilities
        ```

    === "uv (isolated)"

        Use `uv`/`uvx` ([installation](https://docs.astral.sh/uv/getting-started/installation/)) to install & run the `mini` agent in an isolated environment.

        Quickly install + run:

        ```bash
        uvx mini-swe-agent  # simple UI
        uvx mini-swe-agent -v  # visual UI
        uvx --from mini-swe-agent mini-extra  # extra utilities
        ```

        Permanently install

        ```bash
        uv tool install mini-swe-agent
        # then
        mini  # simple UI
        mini -v  # visual UI
        mini-extra  # extra utilities
        ```

    === "pipx (isolated)"

        Use pipx ([installation](https://pipx.pypa.io/stable/installation/)) to install & run `mini` in an isolated environment.

        Quick install + run:

        ```bash
        # Simple UI
        pipx run mini-swe-agent
        # Textual UI
        pipx run mini-swe-agent -v
        # Extra utilities
        pipx run --spec mini-swe-agent mini-extra
        ```

        or for a persistent installation (recommended):

        ```bash
        pipx install mini-swe-agent
        # then
        mini  # simple UI
        mini -v  # visual UI
        mini-extra  # extra utilities
        ```

        If the invocation doesn't immediately work, you might need to run `pipx ensurepath`.

    === "From source/dev"

        For development or if you want to customize the agent:

        ```bash
        git clone https://github.com/SWE-agent/mini-swe-agent.git
        cd mini-swe-agent
        pip install -e .
        ```

        Then run:

        ```bash
        mini  # simple UI
        mini -v  # visual UI
        mini-extra  # extra utilities
        ```

        Or pick a [run script](https://github.com/SWE-agent/mini-swe-agent/tree/main/src/minisweagent/run):

        ```bash
        python minisweagent/run/hello_world.py
        ```

        If you are planning to contribute, please also install the dev dependencies
        and `pre-commit` hooks:

        ```bash
        pip install -e '.[dev]'
        pip install pre-commit && pre-commit install
        ```

        To check your installation, you can run `pytest -n auto` in the root folder.
        This should run all tests in parallel (should take ~3min to run).

        Note that there are still some extra dependencies that are not installed by default
        (basically anything that is in an `.../extra/...` folder).
        If you truly want to get the maximal package, you can run `pip install -e '.[full]'`

!!! note "Changelog"

    Please see the [github release notes](https://github.com/SWE-agent/mini-swe-agent/releases) for recent changes.

!!! example "Example Prompts"

    Try mini-SWE-agent with these example prompts:

    - Implement a Sudoku solver in python in the `sudoku` folder. Make sure the codebase is modular and well tested with pytest.
    - Please run pytest on the current project, discover failing unittests and help me fix them. Always make sure to test the final solution.
    - Help me document & type my codebase by adding short docstrings and type hints.

## Models

!!! tip "TLDR: Models should be set up the first time you run `mini`"

    If you missed the setup wizard, just run `mini-extra config setup`, or take a look at the following section.
    If you want to use local models, please check this [guide](advanced/local_models.md).

### Setting API keys

There are several ways to set your API keys:

* Recommended: Run our setup script: `mini-extra config setup`. This should also run automatically the first time you run `mini`.
* Use `mini-extra config set ANTHROPIC_API_KEY <your-api-key>` to put the key in the `mini` [config file](advanced/configuration.md).
* Export your key as an environment variable: `export ANTHROPIC_API_KEY=<your-api-key>` (this is not persistent if you restart your shell, unless you add it to your shell config, like `~/.bashrc` or `~/.zshrc`).
* If you only use a single model, you can also set `MSWEA_MODEL_API_KEY` (as environment variable or in the config file). This takes precedence over all other keys.
* If you run several agents in parallel, see our note about rotating anthropic keys [here](advanced/configuration.md).

### Selecting a model

* All command line interfaces allow you to set the model name with `-m` or `--model`.
* `mini-extra config setup` can set the default model for you
* In addition, you can set the default model with `mini-extra config set MSWEA_MODEL_NAME <model-name>`, by editing the global [config file](advanced/configuration.md) (shortcut: `mini-extra config edit`), or by setting the `MSWEA_MODEL_NAME` environment variable.
* You can also set your model in a config file (key `model_name` under `model`).
* If you want to use local models, please check this [guide](advanced/local_models.md).

{% include-markdown "_footer.md" %}

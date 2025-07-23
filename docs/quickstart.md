# Quick start

!!! tip "Installation Options"

    === "pip"

        Use pip to install `mini` in your current environment:

        ```bash
        pip install mini-swe-agent
        ```

        And try our command line interface

        ```bash
        # Simple UI
        mini
        # Textual UI
        mini -v
        ```

    === "pipx"

        Use pipx to install & run `mini` in an isolated environment.

        First [install pipx](https://pipx.pypa.io/stable/installation/), then run:

        ```bash
        # Simple UI
        pipx run mini-swe-agent
        # Textual UI
        pipx run mini-swe-agent -v
        ```

        If the invocation doesn't immediately work, you might need to run `pipx ensurepath`.

    === "From source"

        For development or if you want to customize the agent:

        ```bash
        git clone https://github.com/SWE-agent/mini-swe-agent.git
        cd mini-swe-agent
        pip install -e .
        ```

        Then run:

        ```bash
        # Simple UI
        mini
        # Textual UI
        mini -v
        ```

        Or pick a [run script](reference/run/):

        ```bash
        python minisweagent/run/hello_world.py
        ```

        If you are planning to contribute, please also install the dev dependencies
        and `pre-commit` hooks:

        ```bash
        pip install -e '.[dev]'
        pre-commit install
        ```

!!! example "Example Prompts"

    Try mini-SWE-agent with these example prompts:

    - Implement a Sudoku solver in python in the `sudoku` folder. Make sure the codebase is modular and well tested with pytest.
    - Please run pytest on the current project, discover failing unittests and help me fix them. Always make sure to test the final solution.
    - Help me document & type my codebase by adding short docstrings and type hints.

## Models

### Setting API keys

There are several ways to set your API kyes:

* Export your key as an environment variable: `export ANTHROPIC_API_KEY=<your-api-key>` (this is not persistent if you restart your shell, unless you add it to your shell config, like `~/.bashrc` or `~/.zshrc`).
* Use `mini-extra config set ANTHROPIC_API_KEY <your-api-key>` to put the key in the mini config file. The location of the config file is printed when you run `mini --help`.
* If you only use a single model, you can also set `MSWEA_MODEL_API_KEY` (as environment variable or in the config file). This takes precedence over all other keys.
* If you run several agents in parallel, see our note about anthropic keys [here](advanced/configuration.md).

### Selecting a model

* All command line interfaces allow you to set the model name with `-m` or `--model`.
* In addition, you can set the default model with `mini-extra config set MSWEA_MODEL_NAME <model-name>` or by editing the global config file.
  If you run `mini` for the first time and do not use the `--model` flag, it will prompt you for the default model.
* You can also set your model in a config file (key `model_name` under `model`).

{% include-markdown "_footer.md" %}

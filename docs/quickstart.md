# Quick start

!!! tip "Installation Options"

    === "pip"

        Use pip to install `micro` in your current environment:

        ```bash
        pip install micro-swe-agent
        ```

        And try our command line interface

        ```bash
        # Simple UI
        micro
        # Textual UI
        micro -v
        ```

    === "pipx"

        Use pipx to install & run `micro` in an isolated environment.

        First [install pipx](https://pipx.pypa.io/stable/installation/), then run:

        ```bash
        # Simple UI
        pipx run --spec git+ssh://git@github.com/SWE-agent/micro-swe-agent micro
        # Textual UI
        pipx run --spec git+ssh://git@github.com/SWE-agent/micro-swe-agent micro -v
        ```

        If the invocation doesn't immediately work, you might need to run `pipx ensurepath`.

    === "From source"

        For development or if you want to customize the agent:

        ```bash
        git clone https://github.com/SWE-agent/micro-swe-agent.git
        cd micro-swe-agent
        pip install -e .
        ```

        Then run:

        ```bash
        # Simple UI
        micro
        # Textual UI
        micro -v
        ```

        Or pick a [run script](reference/run/):

        ```bash
        python microsweagent/run/hello_world.py
        ```

!!! example "Example Prompts"

    Try micro-SWE-agent with these example prompts:

    - Implement a Sudoku solver in python in the `sudoku` folder. Make sure the codebase is modular and well tested with pytest.
    - Please run pytest on the current project, discover failing unittests and help me fix them. Always make sure to test the final solution.
    - Help me document & type my codebase by adding short docstrings and type hints.

## Models

### Setting API keys

There are several ways to set your API kyes:

* Export your key as an environment variable: `export ANTHROPIC_API_KEY=<your-api-key>` (this is not persistent if you restart your shell, unless you add it to your shell config, like `~/.bashrc` or `~/.zshrc`).
* Use `micro-extra config set ANTHROPIC_API_KEY <your-api-key>` to put the key in the micro config file. The location of the config file is printed when you run `micro --help`.
* If you only use a single model, you can also set `MSWEA_MODEL_API_KEY` (as environment variable or in the config file). This takes precedence over all other keys.
* If you run several agents in parallel, see our note about anthropic keys [here](advanced/configuration.md).

### Selecting a model

* All command line interfaces allow you to set the model name with `-m` or `--model`.
* In addition, you can set the default model with `micro-extra config set MSWEA_MODEL_NAME <model-name>` or by editing the global config file.
  If you run `micro` for the first time and do not use the `--model` flag, it will prompt you for the default model.
* You can also set your model in a config file (key `model_name` under `model`).

{% include-markdown "_footer.md" %}

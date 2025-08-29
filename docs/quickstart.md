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
        python src/minisweagent/run/hello_world.py
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

!!! note "Models should be set up the first time you run `mini`"

    If you missed the setup wizard, just run `mini-extra config setup`, or take a look at the following section.
    If you want to use local models, please check this [guide](advanced/local_models.md).
    Please always include the provider in the model name, e.g., `anthropic/claude-...`.

!!! success "Which model to use?"

    We recommend using `anthropic/claude-sonnet-4-20250514` for most tasks.
    For openai models, we recommend using `openai/gpt-5` or `openai/gpt-5-mini`.
    You can check scores of different models at our [SWE-bench (bash-only)](https://swebench.com) leaderboard.

### Setting API keys

There are several ways to set your API keys:

* **Recommended**: Run our setup script: `mini-extra config setup`. This should also run automatically the first time you run `mini`.
* Use `mini-extra config set ANTHROPIC_API_KEY <your-api-key>` to put the key in the `mini` [config file](advanced/configuration.md).
* Export your key as an environment variable: `export ANTHROPIC_API_KEY=<your-api-key>` (this is not persistent if you restart your shell, unless you add it to your shell config, like `~/.bashrc` or `~/.zshrc`).
* If you only use a single model, you can also set `MSWEA_MODEL_API_KEY` (as environment variable or in the config file). This takes precedence over all other keys.
* If you run several agents in parallel, see our note about rotating anthropic keys [here](advanced/configuration.md).

??? note "All the API key names"

    We use [`litellm`](https://github.com/BerriAI/litellm) to support most models.
    Here's a list of all the API key names available in `litellm`:

    ```
    ALEPH_ALPHA_API_KEY
    ALEPHALPHA_API_KEY
    ANTHROPIC_API_KEY
    ANYSCALE_API_KEY
    AZURE_AI_API_KEY
    AZURE_API_KEY
    AZURE_OPENAI_API_KEY
    BASETEN_API_KEY
    CEREBRAS_API_KEY
    CLARIFAI_API_KEY
    CLOUDFLARE_API_KEY
    CO_API_KEY
    CODESTRAL_API_KEY
    COHERE_API_KEY
    DATABRICKS_API_KEY
    DEEPINFRA_API_KEY
    DEEPSEEK_API_KEY
    FEATHERLESS_AI_API_KEY
    FIREWORKS_AI_API_KEY
    FIREWORKS_API_KEY
    FIREWORKSAI_API_KEY
    GEMINI_API_KEY
    GROQ_API_KEY
    HUGGINGFACE_API_KEY
    INFINITY_API_KEY
    MARITALK_API_KEY
    MISTRAL_API_KEY
    NEBIUS_API_KEY
    NLP_CLOUD_API_KEY
    NOVITA_API_KEY
    NVIDIA_NIM_API_KEY
    OLLAMA_API_KEY
    OPENAI_API_KEY
    OPENAI_LIKE_API_KEY
    OPENROUTER_API_KEY
    OR_API_KEY
    PALM_API_KEY
    PERPLEXITYAI_API_KEY
    PREDIBASE_API_KEY
    PROVIDER_API_KEY
    REPLICATE_API_KEY
    TOGETHERAI_API_KEY
    VOLCENGINE_API_KEY
    VOYAGE_API_KEY
    WATSONX_API_KEY
    WX_API_KEY
    XAI_API_KEY
    XINFERENCE_API_KEY
    ```

### Selecting a model

!!! note "Model names and providers."

    We support most models using [`litellm`](https://github.com/BerriAI/litellm).
    You can find a list of their supported models and the [here](https://docs.litellm.ai/docs/providers).
    Please always include the provider in the model name, e.g., `anthropic/claude-...`.

* **Recommended**: `mini-extra config setup` (should be run the first time you run `mini`) can set the default model for you
* All command line interfaces allow you to set the model name with `-m` or `--model`.
* In addition, you can set the default model with `mini-extra config set MSWEA_MODEL_NAME <model-name>`, by editing the global [config file](advanced/configuration.md) (shortcut: `mini-extra config edit`), or by setting the `MSWEA_MODEL_NAME` environment variable.
* You can also set your model in a config file (key `model_name` under `model`).
* If you want to use local models, please check this [guide](advanced/local_models.md).

!!! note "Popular models"

    Here's a few examples of popular models:

    ```
    anthropic/claude-sonnet-4-20250514
    openai/gpt-5
    openai/gpt-5-mini
    gemini/gemini-2.5-pro
    deepseek/deepseek-chat
    ```

    ??? note "List of all supported models"

        Here's a list of all model names supported by `litellm` as of Aug 29th 2025.
        For even more recent models, check the [`model_prices_and_context_window.json` file from litellm](https://github.com/BerriAI/litellm/blob/main/model_prices_and_context_window.json).

        ```
        --8<-- "docs/data/all_models.txt"
        ```

To find the corresponding API key, check the previous section.

### Model trouble shooting

This section has examples of common error messages and how to fix them.

#### Invalid API key

```json
AuthenticationError: litellm.AuthenticationError: geminiException - {
  "error": {
    "code": 400,
    "message": "API key not valid. Please pass a valid API key.",
    "status": "INVALID_ARGUMENT",
    "details": [
      {
        "@type": "type.googleapis.com/google.rpc.ErrorInfo",
        "reason": "API_KEY_INVALID",
        "domain": "googleapis.com",
        "metadata": {
          "service": "generativelanguage.googleapis.com"
        }
      },
      {
        "@type": "type.googleapis.com/google.rpc.LocalizedMessage",
        "locale": "en-US",
        "message": "API key not valid. Please pass a valid API key."
      }
    ]
  }
}
 You can permanently set your API key with `mini-extra config set KEY VALUE`.
```

Double check your API key and make sure it is correct.
You can take a look at all your API keys with `mini-extra config edit`.

#### "Weird" authentication error

If you fail to authenticate but don't see the previous error message,
it might be that you forgot to include the provider in the model name.

For example, this:

```
  File "/Users/.../.virtualenvs/openai/lib/python3.12/site-packages/google/auth/_default.py", line 685, in default
    raise exceptions.DefaultCredentialsError(_CLOUD_SDK_MISSING_CREDENTIALS)
google.auth.exceptions.DefaultCredentialsError: Your default credentials were not found. To set up Application Default Credentials, see
https://cloud.google.com/docs/authentication/external/set-up-adc for more information.
```

happens if you forgot to prefix your gemini model with `gemini/`.

#### Error during cost calculation

```
Exception: This model isn't mapped yet. model=together_ai/qwen/qwen3-coder-480b-a35b-instruct-fp8, custom_llm_provider=together_ai.
Add it here - https://github.com/BerriAI/litellm/blob/main/model_prices_and_context_window.json.
```

`litellm` doesn't know about the cost of your model.
Take a look at the model registry section of the [local models](advanced/local_models.md) guide to add it.

#### Temperature not supported

Some models (like `gpt-5`, `o3` etc.) do not support temperature, however our default config specifies `temperature: 0.0`.
You need to switch to a config file that does not specify temperature, e.g., `mini_no_temp.yaml`.

To do this, add `-c mini_no_temp` to your `mini` command.

We are working on a better solution for this (see [this issue](https://github.com/SWE-agent/mini-swe-agent/issues/488)).

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

!!! success "Which model to use?"

    We recommend using `claude-sonnet-4-20250514` for most tasks.
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

* **Recommended**: `mini-extra config setup` (should be run the first time you run `mini`) can set the default model for you
* All command line interfaces allow you to set the model name with `-m` or `--model`.
* In addition, you can set the default model with `mini-extra config set MSWEA_MODEL_NAME <model-name>`, by editing the global [config file](advanced/configuration.md) (shortcut: `mini-extra config edit`), or by setting the `MSWEA_MODEL_NAME` environment variable.
* You can also set your model in a config file (key `model_name` under `model`).
* If you want to use local models, please check this [guide](advanced/local_models.md).


### GPT-5 and friends <a name="gpt-5"></a>

`gpt-5` and friends are not included in `litellm` yet, so we need to register them manually.
For this, first create the following file:

??? note "model_registry.json"

    ```json
    {
        "gpt-5": {
            "max_tokens": 128000,
            "max_input_tokens": 400000,
            "max_output_tokens": 128000,
            "input_cost_per_token": 1.25e-06,
            "output_cost_per_token": 1e-05,
            "cache_read_input_token_cost": 1.25e-07,
            "litellm_provider": "openai",
            "mode": "chat",
            "supported_endpoints": [
                "/v1/chat/completions",
                "/v1/batch",
                "/v1/responses"
            ],
            "supported_modalities": [
                "text",
                "image"
            ],
            "supported_output_modalities": [
                "text"
            ],
            "supports_pdf_input": true,
            "supports_function_calling": true,
            "supports_parallel_function_calling": true,
            "supports_response_schema": true,
            "supports_vision": true,
            "supports_prompt_caching": true,
            "supports_system_messages": true,
            "supports_tool_choice": true,
            "supports_native_streaming": true,
            "supports_reasoning": true
        },
        "gpt-5-mini": {
            "max_tokens": 128000,
            "max_input_tokens": 400000,
            "max_output_tokens": 128000,
            "input_cost_per_token": 2.5e-07,
            "output_cost_per_token": 2e-06,
            "cache_read_input_token_cost": 2.5e-08,
            "litellm_provider": "openai",
            "mode": "chat",
            "supported_endpoints": [
                "/v1/chat/completions",
                "/v1/batch",
                "/v1/responses"
            ],
            "supported_modalities": [
                "text",
                "image"
            ],
            "supported_output_modalities": [
                "text"
            ],
            "supports_pdf_input": true,
            "supports_function_calling": true,
            "supports_parallel_function_calling": true,
            "supports_response_schema": true,
            "supports_vision": true,
            "supports_prompt_caching": true,
            "supports_system_messages": true,
            "supports_tool_choice": true,
            "supports_native_streaming": true,
            "supports_reasoning": true
        },
        "gpt-5-nano": {
            "max_tokens": 128000,
            "max_input_tokens": 400000,
            "max_output_tokens": 128000,
            "input_cost_per_token": 5e-08,
            "output_cost_per_token": 4e-07,
            "cache_read_input_token_cost": 5e-09,
            "litellm_provider": "openai",
            "mode": "chat",
            "supported_endpoints": [
                "/v1/chat/completions",
                "/v1/batch",
                "/v1/responses"
            ],
            "supported_modalities": [
                "text",
                "image"
            ],
            "supported_output_modalities": [
                "text"
            ],
            "supports_pdf_input": true,
            "supports_function_calling": true,
            "supports_parallel_function_calling": true,
            "supports_response_schema": true,
            "supports_vision": true,
            "supports_prompt_caching": true,
            "supports_system_messages": true,
            "supports_tool_choice": true,
            "supports_native_streaming": true,
            "supports_reasoning": true
        },
        "gpt-5-chat": {
            "max_tokens": 32768,
            "max_input_tokens": 1047576,
            "max_output_tokens": 32768,
            "input_cost_per_token": 5e-06,
            "output_cost_per_token": 2e-05,
            "input_cost_per_token_batches": 2.5e-06,
            "output_cost_per_token_batches": 1e-05,
            "cache_read_input_token_cost": 1.25e-06,
            "litellm_provider": "openai",
            "mode": "chat",
            "supported_endpoints": [
                "/v1/chat/completions",
                "/v1/batch",
                "/v1/responses"
            ],
            "supported_modalities": [
                "text",
                "image"
            ],
            "supported_output_modalities": [
                "text"
            ],
            "supports_pdf_input": true,
            "supports_function_calling": true,
            "supports_parallel_function_calling": true,
            "supports_response_schema": true,
            "supports_vision": true,
            "supports_prompt_caching": true,
            "supports_system_messages": true,
            "supports_tool_choice": true,
            "supports_native_streaming": true
        },
        "gpt-5-chat-latest": {
            "max_tokens": 128000,
            "max_input_tokens": 400000,
            "max_output_tokens": 128000,
            "input_cost_per_token": 1.25e-06,
            "output_cost_per_token": 1e-05,
            "cache_read_input_token_cost": 1.25e-07,
            "litellm_provider": "openai",
            "mode": "chat",
            "supported_endpoints": [
                "/v1/chat/completions",
                "/v1/batch",
                "/v1/responses"
            ],
            "supported_modalities": [
                "text",
                "image"
            ],
            "supported_output_modalities": [
                "text"
            ],
            "supports_pdf_input": true,
            "supports_function_calling": true,
            "supports_parallel_function_calling": true,
            "supports_response_schema": true,
            "supports_vision": true,
            "supports_prompt_caching": true,
            "supports_system_messages": true,
            "supports_tool_choice": true,
            "supports_native_streaming": true,
            "supports_reasoning": true
        },
        "gpt-5-2025-08-07": {
            "max_tokens": 128000,
            "max_input_tokens": 400000,
            "max_output_tokens": 128000,
            "input_cost_per_token": 1.25e-06,
            "output_cost_per_token": 1e-05,
            "cache_read_input_token_cost": 1.25e-07,
            "litellm_provider": "openai",
            "mode": "chat",
            "supported_endpoints": [
                "/v1/chat/completions",
                "/v1/batch",
                "/v1/responses"
            ],
            "supported_modalities": [
                "text",
                "image"
            ],
            "supported_output_modalities": [
                "text"
            ],
            "supports_pdf_input": true,
            "supports_function_calling": true,
            "supports_parallel_function_calling": true,
            "supports_response_schema": true,
            "supports_vision": true,
            "supports_prompt_caching": true,
            "supports_system_messages": true,
            "supports_tool_choice": true,
            "supports_native_streaming": true,
            "supports_reasoning": true
        },
        "gpt-5-mini-2025-08-07": {
            "max_tokens": 128000,
            "max_input_tokens": 400000,
            "max_output_tokens": 128000,
            "input_cost_per_token": 2.5e-07,
            "output_cost_per_token": 2e-06,
            "cache_read_input_token_cost": 2.5e-08,
            "litellm_provider": "openai",
            "mode": "chat",
            "supported_endpoints": [
                "/v1/chat/completions",
                "/v1/batch",
                "/v1/responses"
            ],
            "supported_modalities": [
                "text",
                "image"
            ],
            "supported_output_modalities": [
                "text"
            ],
            "supports_pdf_input": true,
            "supports_function_calling": true,
            "supports_parallel_function_calling": true,
            "supports_response_schema": true,
            "supports_vision": true,
            "supports_prompt_caching": true,
            "supports_system_messages": true,
            "supports_tool_choice": true,
            "supports_native_streaming": true,
            "supports_reasoning": true
        },
        "gpt-5-nano-2025-08-07": {
            "max_tokens": 128000,
            "max_input_tokens": 400000,
            "max_output_tokens": 128000,
            "input_cost_per_token": 5e-08,
            "output_cost_per_token": 4e-07,
            "cache_read_input_token_cost": 5e-09,
            "litellm_provider": "openai",
            "mode": "chat",
            "supported_endpoints": [
                "/v1/chat/completions",
                "/v1/batch",
                "/v1/responses"
            ],
            "supported_modalities": [
                "text",
                "image"
            ],
            "supported_output_modalities": [
                "text"
            ],
            "supports_pdf_input": true,
            "supports_function_calling": true,
            "supports_parallel_function_calling": true,
            "supports_response_schema": true,
            "supports_vision": true,
            "supports_prompt_caching": true,
            "supports_system_messages": true,
            "supports_tool_choice": true,
            "supports_native_streaming": true,
            "supports_reasoning": true
        }
    }
    ```

Now tell `mini` where to find the file, e.g.,

```bash
mini-extra config set LITELLM_MODEL_REGISTRY_PATH $HOME/model_registry.json
```

Now you're good to go! The only thing to keep in mind is to

1. Reference the model together with the provider, e.g., `openai/gpt-5` (rather than just `gpt-5`)
2. Select a config file without temperature setting, e.g., [`mini_no_temp.yaml`](https://github.com/SWE-agent/mini-swe-agent/blob/main/src/minisweagent/config/mini_no_temp.yaml)

Here's a few examples:

=== "GPT-5"

    ```bash
    mini -v -m openai/gpt-5 -c mini_no_temp
    ```

=== "GPT-5-mini"

    ```bash
    mini -v -m openai/gpt-5-mini -c mini_no_temp
    ```

=== "GPT-5-nano"

    ```bash
    mini -v -m openai/gpt-5-nano -c mini_no_temp
    ```

Or with the [visual UI](usage/mini_v.md):

=== "GPT-5"

    ```bash
    mini -v -m openai/gpt-5 -c mini_no_temp
    ```

=== "GPT-5-mini"

    ```bash
    mini -v -m openai/gpt-5-mini -c mini_no_temp
    ```

=== "GPT-5-nano"

    ```bash
    mini -v -m openai/gpt-5-nano -c mini_no_temp
    ```

{% include-markdown "_footer.md" %}

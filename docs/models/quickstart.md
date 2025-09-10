# Model setup quickstart

!!! tip "Setup"

    In most cases, you can simply run `mini-extra config setup` to set up your default model and API keys.
    This should be run the first time you run `mini`.

## Setting API keys

There are several ways to set your API keys:

* **Recommended**: Run our setup script: `mini-extra config setup`. This should also run automatically the first time you run `mini`.
* Use `mini-extra config set ANTHROPIC_API_KEY <your-api-key>` to put the key in the `mini` [config file](../advanced/global_configuration.md).
* Export your key as an environment variable: `export ANTHROPIC_API_KEY=<your-api-key>` (this is not persistent if you restart your shell, unless you add it to your shell config, like `~/.bashrc` or `~/.zshrc`).
* If you only use a single model, you can also set `MSWEA_MODEL_API_KEY` (as environment variable or in the config file). This takes precedence over all other keys.
* If you run several agents in parallel, see our note about rotating anthropic keys [here](../advanced/global_configuration.md).

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

## Selecting a model

!!! note "Model names and providers."

    We support most models using [`litellm`](https://github.com/BerriAI/litellm).
    You can find a list of their supported models [here](https://docs.litellm.ai/docs/providers).
    Please always include the provider in the model name, e.g., `anthropic/claude-...`.

* **Recommended**: `mini-extra config setup` (should be run the first time you run `mini`) can set the default model for you
* All command line interfaces allow you to set the model name with `-m` or `--model`.
* In addition, you can set the default model with `mini-extra config set MSWEA_MODEL_NAME <model-name>`, by editing the global [config file](../advanced/global_configuration.md) (shortcut: `mini-extra config edit`), or by setting the `MSWEA_MODEL_NAME` environment variable.
* You can also set your model in a config file (key `model_name` under `model`).
* If you want to use local models, please check this [guide](local_models.md).

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

--8<-- "docs/_footer.md"


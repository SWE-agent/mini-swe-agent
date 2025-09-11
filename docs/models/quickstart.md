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

## Extra settings

To configure reasoning efforts or similar settings, you need to edit the [agent config file](../advanced/yaml_configuration.md).

Here's a few examples:

=== "Temperature"

    ```yaml
    model:
      model_kwargs:
        model_name: "anthropic/claude-sonnet-4-20250514"
        temperature: 0.0
    ```

    Note that temperature isn't supported by all models.

=== "GPT-5 reasoning effort"

    ```yaml
    model:
      model_name: "gpt-5-mini"
      model_kwargs:
        drop_params: true
        reasoning_effort: "high"
        verbosity: "medium"
    ```

=== "OpenRouter"

    This example explicitly sets the model class to `openrouter` (see the next section for more details).
    It also explicitly sets the providers to disable switching between them.

    ```yaml
    model:
        model_name: "moonshotai/kimi-k2-0905"
        model_class: "openrouter"
        model_kwargs:
            temperature: 0.0
            provider:
              allow_fallbacks: false
              only: ["Moonshot AI"]
    ```

=== "Local models"

    ```yaml
    model:
      model_name: "my-local-model"
      model_kwargs:
        custom_llm_provider: "openai"
        api_base: "https://..."
        ...
    ```

    See [this guide](local_models.md) for more details on local models.
    In particular, you need to configure token costs for local models.

## Model classes

We support the various models through different backends.
By default (if you only specify the model name), we pick the best backend for you.
This will almost always default to `litellm` (with Anthropic models being a special case as they need to have explicit cache breakpoint handling).

However, there are a few other backends that you can use and specify with the `--model-class` flag or the
`model.model_class` key in the [agent config file](../advanced/yaml_configuration.md).

* **`litellm`** ([`LitellmModel`](../reference/models/litellm.md)) - **Default and recommended**. Supports most models through [litellm](https://github.com/BerriAI/litellm). Works with OpenAI, Anthropic, Google, and many other providers.

* **`anthropic`** ([`AnthropicModel`](../reference/models/anthropic.md)) - Wrapper around `LitellmModel` for Anthropic models that adds cache breakpoint handling.

* **`openrouter`** ([`OpenRouterModel`](../reference/models/openrouter.md)) - Direct integration with [OpenRouter](https://openrouter.ai/) API for accessing various models through a single endpoint.

On top, there's a few more exotic model classes that you can use:

* **`deterministic`** ([`DeterministicModel`](../reference/models/test_models.md)) - Returns predefined responses for testing and development purposes.
* **`minisweagent.models.extra.roulette.RouletteModel` and `minisweagent.models.extra.roulette.InterleavingModel`** ([`RouletteModel`](../reference/models/extra.md) and [`InterleavingModel`](../reference/models/extra.md)) - Randomly selects or interleaves multiple configured models for each query. See [this blog post](https://www.swebench.com/SWE-bench/blog/2025/08/19/mini-roulette/) for more details.

As with the last two, you can also specify any import path to your own custom model class (even if it is not yet part of the mini-SWE-agent package).

--8<-- "docs/_footer.md"


!!! abstract "Local models"

    * This guide shows how to set up local models.
    * You should already be familiar with the [quickstart guide](../quickstart.md).
    * You should also quickly skim the [global configuration guide](../advanced/global_configuration.md) to understand
      the global configuration and [yaml configuration files guide](../advanced/yaml_configuration.md).


!!! tip "Examples"

    * [Issue #303](https://github.com/SWE-agent/mini-swe-agent/issues/303) has several examples of how to use local models.
    * We also welcome concrete examples of how to use local models per pull request into this guide.

## Using litellm

Currently, all models are supported via [`litellm`](https://www.litellm.ai/)
(but if you have specific needs, we're open to add more specific model classes in the [`models`](https://github.com/SWE-agent/mini-swe-agent/tree/main/src/minisweagent/models) submodule).

If you use local models, you most likely need to add some extra keywords to the `litellm` call.
This is done with the `model_kwargs` dictionary which is directly passed to `litellm.completion`.

In other words, this is how we invoke litellm:

```python
litellm.completion(
    model=model_name,
    messages=messages,
    **model_kwargs
)
```

You can set `model_kwargs` in an agent config file like the following one:

??? note "Default configuration file"

    ```yaml
    --8<-- "src/minisweagent/config/mini.yaml"
    ```

In the last section, you can add

```yaml
model:
  model_name: "my-local-model"
  model_kwargs:
    custom_llm_provider: "openai"
    api_base: "https://..."
    ...
  ...
```

!!! tip "Updating the default `mini` configuration file"

    You can set the `MSWEA_MINI_CONFIG_PATH` setting to set path to the default `mini` configuration file.
    This will allow you to override the default configuration file with your own.
    See the [global configuration guide](../advanced/global_configuration.md) for more details.

If this is not enough, our model class should be simple to modify:

??? note "Complete model class"

    - [Read on GitHub](https://github.com/swe-agent/mini-swe-agent/blob/main/src/minisweagent/models/litellm_model.py)
    - [API reference](../reference/models/litellm.md)

    ```python
    --8<-- "src/minisweagent/models/litellm_model.py"
    ```

The other part that you most likely need to figure out are costs.
There are two ways to do this with `litellm`:

1. You set up a litellm proxy server (which gives you a lot of control over all the LM calls)
2. You update the model registry (next section)

### Updating the model registry

LiteLLM get its cost and model metadata from [this file](https://github.com/BerriAI/litellm/blob/main/model_prices_and_context_window.json). You can override or add data from this file if it's outdated or missing your desired model by including a custom registry file.

The model registry JSON file should follow LiteLLM's format:

```json
{
  "my-custom-model": {
    "max_tokens": 4096,
    "input_cost_per_token": 0.0001,
    "output_cost_per_token": 0.0002,
    "litellm_provider": "openai",
    "mode": "chat"
  },
  "my-local-model": {
    "max_tokens": 8192,
    "input_cost_per_token": 0.0,
    "output_cost_per_token": 0.0,
    "litellm_provider": "ollama",
    "mode": "chat"
  }
}
```

!!! warning "Model names"

    Model names are case sensitive. Please make sure you have an exact match.

There are two ways of setting the path to the model registry:

1. Set `LITELLM_MODEL_REGISTRY_PATH` (e.g., `mini-extra config set LITELLM_MODEL_REGISTRY_PATH /path/to/model_registry.json`)
2. Set `litellm_model_registry` in the agent config file

```yaml
model:
  litellm_model_registry: "/path/to/model_registry.json"
  ...
...
```

## Concrete examples

!!! success "Help us fill this section!"

    We welcome concrete examples of how to use local models per pull request into this guide.
    Please add your example here.

--8<-- "docs/_footer.md"

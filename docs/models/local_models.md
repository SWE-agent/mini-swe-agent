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

### Setup local vllm server
To set up local vllm server and run mini-swe-agent, we can either setup a litellm proxy server, or, we can directly set up vllm server then use LiteLLM to hook it to mini-swe-agent infra.   
Take Qwen/Qwen3-Coder-30B-A3B-Instruct as an example. After we spin up the vllm server, first, we update the model registry. We can create a `model_registry.json` file at the root directory of the repo, and add:
```python
"hosted_vllm/Qwen/Qwen3-Coder-30B-A3B-Instruct-FP8": {
   "max_tokens": 132000,
   "input_cost_per_token": 5e-08,
   "output_cost_per_token": 4e-07,
   "litellm_provider": "hosted_vllm",
   "mode": "chat"
}
```
You can source it like this: `LITELLM_MODEL_REGISTRY_PATH="/path/to/your/model/registry.json"`
Then we can create a new yaml file at `src/minisweagent/config/extra/', let's name it `swebench_qwen.yaml`. We can copy the template in `swebench.yaml` and add:
```python
  model:
    model_name: hosted_vllm/Qwen/Qwen3-Coder-30B-A3B-Instruct
    litellm_model_registry: path/to/model_registry.json
  model_kwargs:
    custom_llm_provider: hosted_vllm
    api_base: http://localhost:8000/v1
    api_key: EMPTY

```
Note that the format and variable names in this section are different in litellm's doc, and we should follow what are illustrated in mini-swe-agent's doc.   

Then we need to add vllm hosted model information into [this file](https://github.com/BerriAI/litellm/blob/main/model_prices_and_context_window.json). When we setup mini-swe-agent, LiteLLM should have been installed. If not, run `pip install litellm`, then go into environment's `site-packages`, find LiteLLM and locate this file, add: 
```python
"hosted_vllm/Qwen/Qwen3-Coder-30B-A3B-Instruct": {
       "max_tokens": 1320000,
       "max_input_tokens": 1310000,
       "max_output_tokens": 4096,
       "input_cost_per_token": 5e-08,
       "output_cost_per_token": 4e-07,
       "litellm_provider": "hosted_vllm",
       "mode": "chat",
       "supports_tool_choice": true
   }

```
We can set up the cost here.   
After these steps, we can run the commands to run mini-swe agent on the local server.
    

--8<-- "docs/_footer.md"

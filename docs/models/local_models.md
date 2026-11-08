!!! abstract "Local models"

    * This guide shows how to set up local models.
    * You should already be familiar with the [quickstart guide](../quickstart.md).
    * You should also quickly skim the [global configuration guide](../advanced/global_configuration.md) to understand
      the global configuration and [yaml configuration files guide](../advanced/yaml_configuration.md).


!!! tip "Examples"

    * [Issue #303](https://github.com/SWE-agent/mini-swe-agent/issues/303) has several examples of how to use local models.
    * We also welcome concrete examples of how to use local models per pull request into this guide.

## Using litellm

Currently, models are supported via [`litellm`](https://www.litellm.ai/) by default.

There are typically two steps to using local models:

1. Editing the agent config file to add settings like `custom_llm_provider` and `api_base`.
2. Either ignoring errors from cost tracking or updating the model registry to include your local model.

### Setting API base/provider

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

### Cost tracking

If you run with the above, you will most likely get an error about missing cost information.

If you do not need cost tracking, you can ignore these errors, ideally by editing your agent config file to add:

```yaml
model:
  cost_tracking: "ignore_errors"
  ...
...
```

Alternatively, you can set the global setting:

```bash
export MSWEA_COST_TRACKING="ignore_errors"
```

However, note that this is a global setting, and will affect all models!

However, the best way to handle the cost issue is to add a model registry to litellm to include your local model.

LiteLLM gets its cost and model metadata from [this file](https://github.com/BerriAI/litellm/blob/main/model_prices_and_context_window.json). You can override or add data from this file if it's outdated or missing your desired model by including a custom registry file.

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

!!! warning "Model provider"

    If you use the `custom_llm_provider` or have a provider prefixed to the model name (e.g., `openai/...`),
    then this must also match `litellm_provider` in the config!

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

### Running local models with llama-server (Windows PowerShell)

[`llama-server`](https://github.com/ggml-org/llama.cpp/tree/master/tools/server)
provides an OpenAI-compatible API that mini-SWE-agent can use through LiteLLM.
This example was smoke-tested on Windows 11 using mini-SWE-agent 2.4.6,
LiteLLM 1.105.1, and Qwen3.5-0.8B (Q8_0). It verifies connectivity and shell
execution, not the model's ability to solve coding tasks autonomously.

Start the model server in a separate PowerShell terminal:

```powershell
llama.exe serve `
    -hf ggml-org/Qwen3.5-0.8B-GGUF `
    --alias local-coder `
    --host 127.0.0.1 `
    --port 9931 `
    --ctx-size 8192 `
    --jinja `
    --reasoning off
```

Alternatively, load your own GGUF file with `-m PATH_TO_MODEL.gguf`. The
`--alias` must match the model ID used by mini-SWE-agent. The `--jinja` option
is useful for tool calling. For the tested Qwen model, `--reasoning off`
prevented replies containing only `reasoning_content` and no final `content`.
Other model/template combinations may behave differently. Do not expose an
unauthenticated server to untrusted networks.

Check model discovery in another PowerShell terminal:

```powershell
(Invoke-RestMethod 'http://127.0.0.1:9931/v1/models').data | Select-Object id
```

Create `llama-server.yaml` in your working directory:

```yaml
model:
  model_name: "openai/local-coder"
  cost_tracking: "ignore_errors"
  model_kwargs:
    api_base: "http://127.0.0.1:9931/v1"
    api_key: "sk-local"
```

The `openai/` prefix selects LiteLLM's OpenAI-compatible provider; the API
key is a placeholder for this unauthenticated local server. If you enable
server-side authentication, use the configured key instead. The
`cost_tracking` setting ignores missing local-model price metadata.

Run a harmless connectivity test from a dedicated working directory:

```powershell
mini --config mini.yaml --config .\llama-server.yaml `
    --task "Run echo MINI_LLAMA_OK, verify its output, then finish using echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT. Do not modify files."
```

When using `--config`, explicitly load `mini.yaml` as shown: passing custom
configuration disables automatic loading of the default YAML file. On first
run, the global setup wizard may ask for a default model. Enter
`openai/local-coder`; the API key prompt can be skipped when the key is
supplied in YAML. Inspect and approve each command before execution.

The tested setup executed `echo MINI_LLAMA_OK` with exit code 0, passed its
output back to the model, and reached the interactive finish prompt after
corrective user feedback. The 0.8B model initially tried Unix-specific
commands; native Windows may use `cmd.exe` rather than Bash. WSL2 is a better
fit for complex Bash-oriented coding tasks.

Troubleshooting:

- No model is available: Load a GGUF model with `-hf` or `-m`; verify
  `/v1/models`, port `9931`, and the exact model alias.
- No requests reach the server: Check the address and port in
  `api_base`, including the `/v1` suffix, and verify that the override YAML
  was loaded.
- Empty `content` and long `reasoning_content`: For supported models,
  try `--reasoning off`. Increasing `max_tokens` alone may not help.
- `ModuleNotFoundError: No module named 'mcp'`: A LiteLLM dependency
  regression has been reported in [mini-SWE-agent issue #1003](https://github.com/SWE-agent/mini-swe-agent/issues/1003).
  In the tested environment, `python -m pip install "litellm==1.105.1"`
  worked around this issue; prefer a fixed release when available rather
  than pinning every installation to this version.
- Missing or malformed tool calls: Confirm that the model and chat
  template support OpenAI-compatible tool calling. A successful plain chat
  request alone does not verify tool-call compatibility.

### Generating SWE-bench trajectories with vLLM

This example shows how to generate SWE-bench trajectories using [vLLM](https://docs.vllm.ai/en/latest/) as the local inference engine.

First, launch a vLLM server with your chosen model. For example:

```bash
vllm serve ricdomolm/mini-coder-1.7b &
```

By default, the server will be available at `http://localhost:8000`.

Second, edit the mini-swe-agent SWE-bench config file located in `src/minisweagent/config/benchmarks/swebench.yaml` to include your local vLLM model:

```yaml
model:
  model_name: "hosted_vllm/ricdomolm/mini-coder-1.7b"  # or hosted_vllm/path/to/local/model
  model_kwargs:
    api_base: "http://localhost:8000/v1"  # adjust if using a non-default port/address
```

If you need a custom registry, as detailed above, create a `registry.json` file:

```bash
cat > registry.json <<'EOF'
{
  "ricdomolm/mini-coder-1.7b": {
    "max_tokens": 40960,
    "input_cost_per_token": 0.0,
    "output_cost_per_token": 0.0,
    "litellm_provider": "hosted_vllm",
    "mode": "chat"
  }
}
EOF
```

Now you’re ready to generate trajectories! Let's solve the `django__django-11099` instance of SWE-bench Verified:

```bash
LITELLM_MODEL_REGISTRY_PATH=registry.json mini-extra swebench \
    --output test/ --subset verified --split test --filter '^(django__django-11099)$'
```

You should now see the generated trajectory in the `test/` directory.

--8<-- "docs/_footer.md"

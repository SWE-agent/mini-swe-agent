# Model trouble shooting

This section has examples of common error messages and how to fix them.

## Litellm

`litellm` is the default model class and is used to support most models.

### Invalid API key

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

### "Weird" authentication error

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

### Error during cost calculation

```
Exception: This model isn't mapped yet. model=together_ai/qwen/qwen3-coder-480b-a35b-instruct-fp8, custom_llm_provider=together_ai.
Add it here - https://github.com/BerriAI/litellm/blob/main/model_prices_and_context_window.json.
```

`litellm` doesn't know about the cost of your model.
Take a look at the model registry section of the [local models](local_models.md) guide to add it.

Another common mistake is to not include any or the correct provider in the model name (e.g., `gemini-2.0-flash` instead of `gemini/gemini-2.0-flash`).

## Temperature not supported

Some models (like `o1`, `o3`, `GPT-5` etc.) do not support temperature. The default config no longer specifies a temperature value, so this should work out of the box now.

## Portkey

### Error during cost calculation

We use `litellm` to calculate costs for Portkey models because Portkey doesn't seem to provide per-request cost information without
very inconvenient APIs.

This can lead to errors likethis:

```
  File "/opt/miniconda3/envs/clash/lib/python3.10/site-packages/minisweagent/models/portkey_model.py", line 85, in query
    cost = litellm.cost_calculator.completion_cost(response)
  File "/opt/miniconda3/envs/clash/lib/python3.10/site-packages/litellm/cost_calculator.py", line 973, in completion_cost
    raise e
  File "/opt/miniconda3/envs/clash/lib/python3.10/site-packages/litellm/cost_calculator.py", line 966, in completion_cost
    raise e
  File "/opt/miniconda3/envs/clash/lib/python3.10/site-packages/litellm/cost_calculator.py", line 928, in completion_cost
    ) = cost_per_token(
  File "/opt/miniconda3/envs/clash/lib/python3.10/site-packages/litellm/cost_calculator.py", line 218, in cost_per_token
    _, custom_llm_provider, _, _ = litellm.get_llm_provider(model=model)
  File "/opt/miniconda3/envs/clash/lib/python3.10/site-packages/litellm/litellm_core_utils/get_llm_provider_logic.py", line 395, in get_llm_provider
    raise e
  File "/opt/miniconda3/envs/clash/lib/python3.10/site-packages/litellm/litellm_core_utils/get_llm_provider_logic.py", line 372, in get_llm_provider
    raise litellm.exceptions.BadRequestError(  # type: ignore
litellm.exceptions.BadRequestError: litellm.BadRequestError: LLM Provider NOT provided. Pass in the LLM provider you are trying to call. You passed model=grok-code-fast-1
 Pass model as E.g. For 'Huggingface' inference endpoints pass in `completion(model='huggingface/starcoder',..)` Learn more: https://docs.litellm.ai/docs/providers
```

In this case, the issue is simply that the portkey model name doesn't match the litellm model name (and very specifically here
doesn't include the provider).

To fix this, you can manually set the litellm model name to the portkey model name with the `litellm_model_name_override` key.
For example:

```yaml
model:
  model_name: "grok-code-fast-1"  # the portkey model name
  model_class: "portkey"  # make sure to use the portkey model class
  litellm_model_name_override: "xai/grok-code-fast-1"  # the litellm model name for cost information
  ...
```

--8<-- "docs/_footer.md"

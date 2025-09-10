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

### Temperature not supported

Some models (like `gpt-5`, `o3` etc.) do not support temperature, however our default config specifies `temperature: 0.0`.
You need to switch to a config file that does not specify temperature, e.g., `mini_no_temp.yaml`.

To do this, add `-c mini_no_temp` to your `mini` command.

We are working on a better solution for this (see [this issue](https://github.com/SWE-agent/mini-swe-agent/issues/488)).

--8<-- "docs/_footer.md"

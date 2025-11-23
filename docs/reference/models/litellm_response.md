# Litellm Response API Model

!!! note "LiteLLM Response API Model class"

    - [Read on GitHub](https://github.com/swe-agent/mini-swe-agent/blob/main/src/minisweagent/models/litellm_response_api_model.py)

    ??? note "Full source code"

        ```python
        --8<-- "src/minisweagent/models/litellm_response_api_model.py"
        ```

!!! tip "When to use this model"

    * Use this model class when you want to use OpenAI's [Responses API](https://platform.openai.com/docs/api-reference/responses) (previously called the Chat Completions API with streaming enabled).
    * This is particularly useful for models like GPT-5 that benefit from the extended thinking/reasoning capabilities provided by the Responses API.

## Usage

To use the Response API model, specify `model_class: "litellm_response"` in your agent config:

```yaml
model:
  model_class: "litellm_response"
  model_name: "openai/gpt-5.1"
  model_kwargs:
    drop_params: true
    reasoning:
      effort: "medium"
    text:
      verbosity: "medium"
```

Or via command line:

```bash
mini -m "openai/gpt-5-mini" --model-class litellm_response
```

::: minisweagent.models.litellm_response_api_model

{% include-markdown "../../_footer.md" %}




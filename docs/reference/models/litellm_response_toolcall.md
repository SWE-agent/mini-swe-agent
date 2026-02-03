# Litellm Response Model

!!! note "LiteLLM Response API Model class"

    - [Read on GitHub](https://github.com/swe-agent/mini-swe-agent/blob/main/src/minisweagent/models/litellm_response_model.py)

    ??? note "Full source code"

        ```python
        --8<-- "src/minisweagent/models/litellm_response_model.py"
        ```

!!! tip "When to use this model"

    * Use this model class when you want to use OpenAI's [Responses API](https://platform.openai.com/docs/api-reference/responses) with native tool calling.
    * This is particularly useful for models like GPT-5 that benefit from the extended thinking/reasoning capabilities provided by the Responses API.
    * This model maintains conversation state across turns using `previous_response_id`.

## Usage

To use the Response API model, specify `model_class: "litellm_response"` in your agent config:

```yaml
model:
  model_class: "litellm_response"
  model_name: "openai/gpt-5.2"
  model_kwargs:
    drop_params: true
    reasoning:
      effort: "high"
```

Or via command line:

```bash
mini -m "openai/gpt-5.2" --model-class litellm_response
```

::: minisweagent.models.litellm_response_model

{% include-markdown "../../_footer.md" %}

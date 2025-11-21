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
    * For most models, the standard [`LitellmModel`](litellm.md) is sufficient and recommended.

## Key Differences from LitellmModel

The `LitellmResponseAPIModel` extends `LitellmModel` with the following changes:

1. **API Endpoint**: Uses `litellm.responses()` instead of `litellm.completion()`
2. **State Management**: Maintains conversation state via `previous_response_id` for multi-turn conversations
3. **Response Parsing**: Implements special parsing logic to extract text from Response API output format

## Usage

To use the Response API model, specify `model_class: "litellm_response"` in your agent config:

```yaml
model:
  model_class: "litellm_response"
  model_name: "openai/gpt-5-mini"
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

## Model Configuration

The Response API supports specific parameters for controlling reasoning and output verbosity:

=== "Basic Configuration"

    ```yaml
    model:
      model_class: "litellm_response"
      model_name: "openai/gpt-5-mini"
      model_kwargs:
        drop_params: true
    ```

=== "With Reasoning Control"

    ```yaml
    model:
      model_class: "litellm_response"
      model_name: "openai/gpt-5-mini"
      model_kwargs:
        drop_params: true
        reasoning:
          effort: "high"  # Options: low, medium, high
        text:
          verbosity: "high"  # Options: low, medium, high
    ```

=== "Temperature Control"

    ```yaml
    model:
      model_class: "litellm_response"
      model_name: "openai/gpt-5"
      model_kwargs:
        drop_params: true
        temperature: 0.7
        reasoning:
          effort: "medium"
    ```

## Supported Models

The Response API model class works with models that support OpenAI's Responses API format:

- `openai/gpt-5`
- `openai/gpt-5-mini`
- Other models that implement the Responses API specification

## Notes

- The `drop_params: true` setting is recommended to automatically filter out parameters not supported by your specific model.
- The model maintains conversation state through `previous_response_id`, allowing for efficient multi-turn conversations.
- Cost tracking works the same way as with the standard `LitellmModel`.

::: minisweagent.models.litellm_response_api_model

{% include-markdown "../../_footer.md" %}


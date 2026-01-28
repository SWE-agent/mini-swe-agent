# Models Overview

This page provides an overview of all available model classes in mini-SWE-agent.

## Model Classes

| Class | Endpoint | Toolcalls | Description |
|-------|----------|-----------|-------------|
| [`LitellmModel`](litellm.md) | `/completion` | ❌ | Default model using [LiteLLM](https://docs.litellm.ai/docs/providers) for broad provider support (OpenAI, Anthropic, 100+ providers) |
| [`LitellmToolcallModel`](litellm_toolcall.md) | `/completion` | ✅ | LiteLLM with native tool calling |
| [`LitellmResponseAPIModel`](litellm_response.md) | `/response` | ❌ | LiteLLM with [OpenAI Responses API](https://platform.openai.com/docs/api-reference/responses) support |
| [`LitellmResponseToolcallModel`](litellm_response.md) | `/response` | ✅ | LiteLLM Responses API with native tool calling |
| [`OpenRouterModel`](openrouter.md) | `/completion` | ❌ | [OpenRouter](https://openrouter.ai/) API integration |
| [`OpenRouterToolcallModel`](openrouter.md) | `/completion` | ✅ | OpenRouter with native tool calling |
| [`OpenRouterResponseAPIToolcallModel`](openrouter.md) | `/response` | ✅ | OpenRouter Responses API with native tool calling |
| [`PortkeyModel`](portkey.md) | `/completion` | ❌ | [Portkey](https://portkey.ai/) AI gateway integration |
| [`PortkeyResponseAPIModel`](portkey_response.md) | `/response` | ❌ | Portkey with Responses API support |
| [`RequestyModel`](requesty.md) | `/completion` | ❌ | [Requesty](https://requesty.ai/) API integration |
| [`DeterministicModel`](test_models.md) | N/A | ❌ | Returns predefined outputs (for testing) |
| [`RouletteModel`](extra.md) | Meta | ❌ | Randomly selects from multiple models |
| [`InterleavingModel`](extra.md) | Meta | ❌ | Alternates between models in sequence |

{% include-markdown "../../_footer.md" %}

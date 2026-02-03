# Models Overview

This page provides an overview of all available model classes in mini-SWE-agent.

## Model Classes

| Class | Shortcut | Endpoint | Toolcalls | Description |
|-------|----------|----------|-----------|-------------|
| [`LitellmModel`](litellm.md) | `litellm` | `/completion` | ✅ | Default model using [LiteLLM](https://docs.litellm.ai/docs/providers) for broad provider support (OpenAI, Anthropic, 100+ providers) |
| [`LitellmTextbasedModel`](litellm.md) | `litellm_textbased` | `/completion` | ❌ | LiteLLM with text-based actions (no native tool calling) |
| [`LitellmResponseModel`](litellm_response_toolcall.md) | `litellm_response` | `/response` | ✅ | LiteLLM with [OpenAI Responses API](https://platform.openai.com/docs/api-reference/responses) and native tool calling |
| [`OpenRouterModel`](openrouter.md) | `openrouter` | `/completion` | ✅ | [OpenRouter](https://openrouter.ai/) API integration |
| [`OpenRouterTextbasedModel`](openrouter.md) | `openrouter_textbased` | `/completion` | ❌ | OpenRouter with text-based actions |
| [`OpenRouterResponseModel`](openrouter.md) | `openrouter_response` | `/response` | ✅ | OpenRouter Responses API with native tool calling |
| [`PortkeyModel`](portkey.md) | `portkey` | `/completion` | ✅ | [Portkey](https://portkey.ai/) AI gateway integration |
| [`PortkeyResponseAPIModel`](portkey_response.md) | `portkey_response` | `/response` | ✅ | Portkey with Responses API support |
| [`RequestyModel`](requesty.md) | `requesty` | `/completion` | ✅ | [Requesty](https://requesty.ai/) API integration |
| [`DeterministicModel`](test_models.md) | `deterministic` | N/A | ❌ | Returns predefined outputs (for testing) |
| [`RouletteModel`](extra.md) | — | Meta | ❌ | Randomly selects from multiple models |
| [`InterleavingModel`](extra.md) | — | Meta | ❌ | Alternates between models in sequence |

{% include-markdown "../../_footer.md" %}

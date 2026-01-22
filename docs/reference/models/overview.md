# Models Overview

This page provides an overview of all available model classes in mini-SWE-agent.

## Model Classes

| Class | Description | Provider |
|-------|-------------|----------|
| [`LitellmModel`](litellm.md) | Default model using LiteLLM for broad provider support | OpenAI, Anthropic, and [100+ providers](https://docs.litellm.ai/docs/providers) |
| [`LitellmResponseAPIModel`](litellm_response.md) | LiteLLM with OpenAI Responses API support | OpenAI |
| [`LitellmToolcallModel`](litellm_toolcall.md) | LiteLLM with native tool calling | OpenAI, Anthropic, etc. |
| [`OpenRouterModel`](openrouter.md) | OpenRouter API integration | [OpenRouter](https://openrouter.ai/) |
| [`RequestyModel`](requesty.md) | Requesty API integration | [Requesty](https://requesty.ai/) |
| [`PortkeyModel`](extra.md) | Portkey AI gateway integration | [Portkey](https://portkey.ai/) |
| [`PortkeyResponseAPIModel`](extra.md) | Portkey with Responses API support | [Portkey](https://portkey.ai/) |
| [`DeterministicModel`](test_models.md) | Returns predefined outputs (for testing) | N/A |
| [`RouletteModel`](extra.md) | Randomly selects from multiple models | Meta |
| [`InterleavingModel`](extra.md) | Alternates between models in sequence | Meta |

{% include-markdown "../../_footer.md" %}

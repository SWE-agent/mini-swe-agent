# API Reference

This section provides detailed documentation for all classes and modules in mini-SWE-agent.

## Agents

- **[DefaultAgent](agents/default.md)** - The minimal default agent implementation
- **[InteractiveAgent](agents/interactive.md)** - Agent with human-in-the-loop functionality
- **[TextualAgent](agents/textual.md)** - Agent with interactive TUI using Textual

## Models

- **[LitellmModel](models/litellm.md)** - Wrapper for LiteLLM models (supports most LLM providers)
- **[LitellmResponseAPIModel](models/litellm_response.md)** - Specialized model for OpenAI's Responses API
- **[AnthropicModel](models/anthropic.md)** - Specialized interface for Anthropic models
- **[DeterministicModel](models/test_models.md)** - Deterministic models for testing
- **[Model Utilities](models/utils.md)** - Convenience functions for model selection and configuration

## Environments

- **[LocalEnvironment](environments/local.md)** - Execute commands in the local environment
- **[DockerEnvironment](environments/docker.md)** - Execute commands in Docker containers
- **[SwerexDockerEnvironment](environments/swerex_docker.md)** - Extended Docker environment with SWE-Rex integration

## Run Scripts

Entry points and command-line interfaces:

- **[Hello World](run/hello_world.md)** - Simple example usage
- **[mini](run/mini.md)** - Interactive local execution
- **[GitHub Issue](run/github_issue.md)** - GitHub issue solver
- **[SWE-bench](run/swebench.md)** - SWE-bench evaluation script

{% include-markdown "../_footer.md" %}
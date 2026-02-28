# Cookbook

!!! abstract "Remixing & extending mini"

    * This guide shows how to mix the different components of the `mini` agent to create your own custom version.
    * You might want to first take a look at the [control flow of the default agent](control_flow.md) first


!!! note "Development setup"

    Make sure to follow the dev setup instructions in [quickstart.md](../quickstart.md).

We provide several different entry points to the agent,
for example [hello world](https://github.com/SWE-agent/mini-swe-agent/blob/main/src/minisweagent/run/hello_world.py),
or the [default when calling `mini`](https://github.com/SWE-agent/mini-swe-agent/blob/main/src/minisweagent/run/mini.py).

Want to cook up your custom version and the config is not enough?
Just follow the recipe below:

1. What's the control flow you need? Pick an [agent class](https://github.com/SWE-agent/mini-swe-agent/blob/main/src/minisweagent/agents) (e.g., [simplest example](https://github.com/SWE-agent/mini-swe-agent/blob/main/src/minisweagent/agents/default.py), [with human in the loop](https://github.com/SWE-agent/mini-swe-agent/blob/main/src/minisweagent/agents/interactive.py))
2. How should actions be executed? Pick an [environment class](https://github.com/SWE-agent/mini-swe-agent/blob/main/src/minisweagent/environments) (e.g., [local](https://github.com/SWE-agent/mini-swe-agent/blob/main/src/minisweagent/environments/local.py), or [docker](https://github.com/SWE-agent/mini-swe-agent/blob/main/src/minisweagent/environments/docker.py))
3. How is the LM queried? Pick a [model class](https://github.com/SWE-agent/mini-swe-agent/blob/main/src/minisweagent/models) (e.g., [litellm](https://github.com/SWE-agent/mini-swe-agent/blob/main/src/minisweagent/models/litellm_model.py))
4. How to invoke the agent? Bind them all together in a [run script](https://github.com/SWE-agent/mini-swe-agent/blob/main/src/minisweagent/run), possibly reading from a [config](https://github.com/SWE-agent/mini-swe-agent/blob/main/src/minisweagent/config) (e.g., [hello world](https://github.com/SWE-agent/mini-swe-agent/blob/main/src/minisweagent/run/hello_world.py), or [`mini` entry point](https://github.com/SWE-agent/mini-swe-agent/blob/main/src/minisweagent/run/mini.py))

We aim to keep all of these components very simple, but offer lots of choice between them -- enough to cover a broad range of
things that you might want to do.

You can override the default entry point by setting the `MSWEA_DEFAULT_RUN` environment variable to the import path of your run script.

## Mix & match

### Models

=== "Hello world (use automatic model selection)"

    ```python
    from minisweagent.agents.default import DefaultAgent
    from minisweagent.models import get_model
    from minisweagent.environments.local import LocalEnvironment

    model_name = "anthropic/claude-sonnet-4-5-20250929"

    agent = DefaultAgent(
        get_model(input_model_name=model_name),
        LocalEnvironment(),
    )
    agent.run(task)
    ```

=== "Hello world (Anthropic)"

    ```python
    from minisweagent.agents.default import DefaultAgent
    from minisweagent.models.anthropic_model import AnthropicModel
    from minisweagent.environments.local import LocalEnvironment

    model_name = "anthropic/claude-sonnet-4-5-20250929"

    agent = DefaultAgent(
        AnthropicModel(model_name=model_name),
        LocalEnvironment(),
    )
    agent.run(task)
    ```

=== "Hello world (Litellm)"

    ```python
    from minisweagent.agents.default import DefaultAgent
    from minisweagent.models.litellm_model import LitellmModel
    from minisweagent.environments.local import LocalEnvironment

    model_name = "gpt-4o"

    agent = DefaultAgent(
        LitellmModel(model_name=model_name),
        LocalEnvironment(),
    )
    agent.run(task)
    ```

### Environments

=== "Hello world with local execution"

    ```python
    from minisweagent.environments.local import LocalEnvironment

    agent = DefaultAgent(
        LitellmModel(model_name=model_name),
        LocalEnvironment(),
    )
    ```

=== "Hello world with docker execution"

    ```python
    from minisweagent.environments.docker import DockerEnvironment

    agent = DefaultAgent(
        LitellmModel(model_name=model_name),
        DockerEnvironment(),
    )
    ```

### Agents

=== "Default agent"

    ```python
    from minisweagent.agents.default import DefaultAgent
    from minisweagent.models import get_model
    from minisweagent.environments.local import LocalEnvironment

    agent = DefaultAgent(
        get_model(input_model_name=model_name),
        LocalEnvironment(),
    )
    ```

=== "Human in the loop"

    ```python
    from minisweagent.agents.interactive import InteractiveAgent
    from minisweagent.models import get_model
    from minisweagent.environments.local import LocalEnvironment

    agent = InteractiveAgent(
        LitellmModel(model_name=model_name),
        LocalEnvironment(),
    )
    ```

=== "Human in the loop (textual)"

    ```python
    from minisweagent.agents.interactive_textual import TextualAgent
    from minisweagent.models import get_model
    from minisweagent.environments.local import LocalEnvironment

    agent = TextualAgent(
        LitellmModel(model_name=model_name),
        LocalEnvironment(),
    )
    ```

## Advanced

### Customizing execution

An agent that uses python function for some actions:


=== "Subclassing the agent"

    ```python
    from minisweagent.agents.default import DefaultAgent
    import shlex

    def python_function(*args) -> dict:
        ...
        return {"output": "..."}

    class AgentWithPythonFunctions(DefaultAgent):
        def execute_action(self, action: dict) -> dict:
            if action["action"].startswith("python_function"):
                args = shlex.split(action["action"].removeprefix("python_function").strip())
                return python_function(*args)
            return super().execute_action(action)
    ```


=== "Subclassing the environment"

    ```python
    from minisweagent.agents.default import DefaultAgent
    import shlex

    def python_function(*args) -> dict:
        ...
        return {"output": "..."}

    class EnvironmentWithPythonFunctions(LocalEnvironment):
        def execute(self, command: str, cwd: str = "") -> dict:
            if command.startswith("python_function"):
                args = shlex.split(command.removeprefix("python_function").strip())
                return python_function(*args)
            return super().execute(command, cwd)

    agent = DefaultAgent(
        LitellmModel(model_name=model_name),
        EnvironmentWithPythonFunctions(),
    )
    ```

An agent that exits when the `submit` command is issued:

=== "Subclassing the agent"

    ```python
    from minisweagent.agents.default import DefaultAgent, Submitted

    class AgentQuitsOnSubmit(DefaultAgent):
        def execute_action(self, action: dict) -> dict:
            if action["action"] == "submit":
                # The `Submitted` exception will be caught by the agent and
                # the final output will be printed.
                raise Submitted("The agent has finished its task.")
            return super().execute_action(action)
    ```

=== "Subclassing the environment"

    ```python
    from minisweagent.agents.default import DefaultAgent, Submitted
    from minisweagent.environments.local import LocalEnvironment

    class EnvironmentQuitsOnSubmit(LocalEnvironment):
        def execute(self, command: str, cwd: str = "") -> dict:
            if command == "submit":
                raise Submitted("The agent has finished its task.")
            return super().execute(command, cwd)

    agent = DefaultAgent(
        LitellmModel(model_name=model_name),
        EnvironmentQuitsOnSubmit(),
    )
    ```


An agent that validates actions before execution (also an example of how to use an extended config class):

=== "Subclassing the agent"

    ```python
    import re
    from dataclasses import dataclass
    from minisweagent.agents.default import (
        DefaultAgent, NonTerminatingException, DefaultAgentConfig
    )

    @dataclass
    class ValidatingAgentConfig(DefaultAgentConfig):
        forbidden_patterns: list[str] = [
            r"rm -rf /",
            r"sudo.*passwd",
            r"mkfs\.",
        ]

    class ValidatingAgent(DefaultAgent):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs, config_class=ValidatingAgentConfig)

        def execute_action(self, action: dict) -> dict:
            for pattern in self.config.forbidden_patterns:
                if re.search(pattern, action["action"], re.IGNORECASE):
                    raise NonTerminatingException("Action blocked")
            return super().execute_action(action)
    ```

=== "Subclassing the environment"

    ```python
    import re
    from dataclasses import dataclass
    from minisweagent.agents.default import (
        DefaultAgent, NonTerminatingException, DefaultAgentConfig
    )
    from minisweagent.environments.local import LocalEnvironment

    @dataclass
    class EnvironmentWithForbiddenPatternsConfig(LocalEnvironmentConfig):
        forbidden_patterns: list[str] = [
            r"rm -rf /",
            r"sudo.*passwd",
            r"mkfs\.",
        ]

    class EnvironmentWithForbiddenPatterns(LocalEnvironment):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs, config_class=EnvironmentWithForbiddenPatternsConfig)

        def execute(self, command: str, cwd: str = "") -> dict:
            for pattern in self.config.forbidden_patterns:
                if re.search(pattern, command, re.IGNORECASE):
                    raise NonTerminatingException("Action blocked")
            return super().execute(command, cwd)

    agent = DefaultAgent(
        LitellmModel(model_name=model_name),
        EnvironmentWithForbiddenPatterns(),
    )
    ```

### Running mini-swe-agent on Ray

[This blog post](https://www.anyscale.com/blog/massively-parallel-agentic-simulations-with-ray)
describes how to parallelize mini-swe-agent runs with Ray.

{% include-markdown "_footer.md" %}

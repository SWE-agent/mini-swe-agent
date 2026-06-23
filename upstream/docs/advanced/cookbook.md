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

## Hello world

See [Python bindings](../usage/python_bindings.md) for the most basic example.

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
        get_model(input_model_name=model_name),
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
        def execute_actions(self, message: dict) -> list[dict]:
            for action in message.get("extra", {}).get("actions", []):
                command = action.get("command", "")
                if command.startswith("python_function"):
                    args = shlex.split(command.removeprefix("python_function").strip())
                    return self.add_messages(self.model.format_observation_messages(
                        message, [python_function(*args)], self.get_template_vars()
                    ))
            # everything else works as usual
            return super().execute_actions(message)
    ```


=== "Subclassing the environment"

    ```python
    from minisweagent.agents.default import DefaultAgent
    from minisweagent.environments.local import LocalEnvironment
    import shlex

    def python_function(*args) -> dict:
        ...
        return {"output": "..."}

    class EnvironmentWithPythonFunctions(LocalEnvironment):
        def execute(self, action: dict, cwd: str = "") -> dict:
            command = action.get("command", "")
            if command.startswith("python_function"):
                args = shlex.split(command.removeprefix("python_function").strip())
                return python_function(*args)
            # all other commands are executed as usual
            return super().execute(action, cwd)

    agent = DefaultAgent(
        LitellmModel(model_name=model_name),
        EnvironmentWithPythonFunctions(),
    )
    ```

An agent that exits when the `submit` command is issued:

=== "Subclassing the agent"

    ```python
    from minisweagent.agents.default import DefaultAgent
    from minisweagent.exceptions import Submitted

    class AgentQuitsOnSubmit(DefaultAgent):
        def execute_actions(self, message: dict) -> list[dict]:
            for action in message.get("extra", {}).get("actions", []):
                if action.get("command", "") == "submit":
                    # The `Submitted` exception will be caught by the agent and
                    # the final output will be printed.
                    raise Submitted({
                        "role": "exit",
                        "content": "The agent has finished its task.",
                        "extra": {"exit_status": "Submitted", "submission": ""},
                    })
            return super().execute_actions(message)
    ```

=== "Subclassing the environment"

    ```python
    from minisweagent.agents.default import DefaultAgent
    from minisweagent.environments.local import LocalEnvironment
    from minisweagent.exceptions import Submitted

    class EnvironmentQuitsOnSubmit(LocalEnvironment):
        def execute(self, action: dict, cwd: str = "") -> dict:
            if action.get("command", "") == "submit":
                raise Submitted({
                    "role": "exit",
                    "content": "The agent has finished its task.",
                    "extra": {"exit_status": "Submitted", "submission": ""},
                })
            return super().execute(action, cwd)

    agent = DefaultAgent(
        LitellmModel(model_name=model_name),
        EnvironmentQuitsOnSubmit(),
    )
    ```


An agent that validates actions before execution (also an example of how to use an extended config class):

=== "Subclassing the agent"

    ```python
    import re
    from minisweagent.agents.default import DefaultAgent, AgentConfig
    from minisweagent.exceptions import FormatError
    from pydantic import BaseModel

    class ValidatingAgentConfig(AgentConfig):
        forbidden_patterns: list[str] = [
            r"rm -rf /",
            r"sudo.*passwd",
            r"mkfs\.",
        ]

    class ValidatingAgent(DefaultAgent):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs, config_class=ValidatingAgentConfig)

        def execute_actions(self, message: dict) -> list[dict]:
            for action in message.get("extra", {}).get("actions", []):
                command = action.get("command", "")
                for pattern in self.config.forbidden_patterns:
                    if re.search(pattern, command, re.IGNORECASE):
                        raise FormatError(self.model.format_message(
                            role="user", content="Action blocked: forbidden pattern detected"
                        ))
            return super().execute_actions(message)
    ```

=== "Subclassing the environment"

    ```python
    import re
    from minisweagent.agents.default import DefaultAgent
    from minisweagent.environments.local import LocalEnvironment, LocalEnvironmentConfig
    from minisweagent.models.litellm_model import LitellmModel

    class EnvironmentWithForbiddenPatternsConfig(LocalEnvironmentConfig):
        forbidden_patterns: list[str] = [
            r"rm -rf /",
            r"sudo.*passwd",
            r"mkfs\.",
        ]

    class EnvironmentWithForbiddenPatterns(LocalEnvironment):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs, config_class=EnvironmentWithForbiddenPatternsConfig)

        def execute(self, action: dict, cwd: str = "") -> dict:
            command = action.get("command", "")
            for pattern in self.config.forbidden_patterns:
                if re.search(pattern, command, re.IGNORECASE):
                    return {"output": "Action blocked: forbidden pattern detected", "returncode": 1}
            return super().execute(action, cwd)

    agent = DefaultAgent(
        LitellmModel(model_name=model_name),
        EnvironmentWithForbiddenPatterns(),
    )
    ```

### Running mini-swe-agent on Ray

[This blog post](https://www.anyscale.com/blog/massively-parallel-agentic-simulations-with-ray)
describes how to parallelize mini-swe-agent runs with Ray.

{% include-markdown "_footer.md" %}

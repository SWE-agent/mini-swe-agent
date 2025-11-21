# Yaml config files

!!! abstract "Agent configuration files"

    * You can configure the agent's behavior using YAML configuration files. This guide shows how to do that.
    * You should already be familiar with the [quickstart guide](../quickstart.md).
    * For global environment settings (API keys, default model, etc., basically anything that can be set as environment variables), see [global configuration](global_configuration.md).
    * Want more? See [python bindings](cookbook.md) for subclassing & developing your own agent.

## Overall structure

Configuration files look like this:

??? note "Configuration file"

    ```yaml
    --8<-- "src/minisweagent/config/mini.yaml"
    ```

We use the following top-level keys:

- `agent`: Agent configuration (prompt templates, cost limits etc.)
- `environment`: Environment configuration (if you want to run in a docker container, etc.)
- `model`: Model configuration (model name, reasoning strength, etc.)
- `run`: Run configuration (output file, etc.)

## Agent configuration

Different agent classes might have slightly different configuration options.
You can find the full list of options in the [API reference](../reference/agents/default.md).

To use a different agent class, you can set the `agent_class` key to the name of the agent class you want to use
or even to an import path (to use your own custom agent class even if it is not yet part of the mini-SWE-agent package).

### Prompt templates

We use [Jinja2](https://jinja.palletsprojects.com/) to render templates (e.g., the instance template).

TL;DR: You include variables with double (!) curly braces, e.g. `{{task}}` to include the task that was given to the agent.

However, you can also do fairly complicated logic like this directly from your template:

??? note "Example: Dealing with long observations"

    The following snippets shortens long observations and displays a warning if the output is too long.

    ```jinja
    <returncode>{{output.returncode}}</returncode>
    {% if output.output | length < 10000 -%}
        <output>
            {{ output.output -}}
        </output>
    {%- else -%}
        <warning>
            The output of your last command was too long.
            Please try a different command that produces less output.
            If you're looking at a file you can try use head, tail or sed to view a smaller number of lines selectively.
            If you're using grep or find and it produced too much output, you can use a more selective search pattern.
            If you really need to see something from the full command's output, you can redirect output to a file and then search in that file.
        </warning>

        {%- set elided_chars = output.output | length - 10000 -%}

        <output_head>
            {{ output.output[:5000] }}
        </output_head>

        <elided_chars>
            {{ elided_chars }} characters elided
        </elided_chars>

        <output_tail>
            {{ output.output[-5000:] }}
        </output_tail>
    {%- endif -%}
    ```

In all builtin agents, you can use the following variables:

- Environment variables (`LocalEnvironment` only, see discussion [here](https://github.com/SWE-agent/mini-swe-agent/pull/425))
- Agent config variables (i.e., anything that was set in the `agent` section of the config file, e.g., `step_limit`, `cost_limit`, etc.)
- Environment config variables (i.e., anything that was set in the `environment` section of the config file, e.g., `cwd`, `timeout`, etc.)
- Variables passed to the `run` method of the agent (by default that's only `task`, but you can pass other variables if you want to)
- Output of the last action execution (i.e., `output` from the `execute_action` method)

### Custom Action Parsing

By default, mini-SWE-agent parses actions from markdown code blocks (` ```bash...``` `).
You can customize this behavior by setting the `action_regex` field to support different formats like XML.

!!! note "Important"

    If you set a custom action_regex (e.g. `<action>(.*?)</action>`), you must use the same output format across all prompt templates (system_template, instance_template, format_error_template, etc.), ensuring the LLM wraps commands accordingly. See the example below for a complete configuration.


??? example "Using XML format instead of markdown"

    This example uses the same structure as the default mini.yaml config, but with `<action>` tags instead of markdown code blocks:

    ```yaml
    agent:
      action_regex: "<action>(.*?)</action>"
      system_template: |
        You are a helpful assistant that can interact with a computer.

        Your response must contain exactly ONE command in <action> tags (instead of markdown).
        Include a THOUGHT section before your command where you explain your reasoning process.
        Format your response as shown in <format_example>.

        <format_example>
        Your reasoning and analysis here. Explain why you want to perform the action.

        <action>your_command_here</action>
        </format_example>

        Failure to follow these rules will cause your response to be rejected.

      instance_template: |
        Please solve this issue: {{task}}

        You can execute bash commands and edit files to implement the necessary changes.

        ## Recommended Workflow

        This workflow should be done step-by-step so that you can iterate on your changes and any possible problems.

        1. Analyze the codebase by finding and reading relevant files
        2. Create a script to reproduce the issue
        3. Edit the source code to resolve the issue
        4. Verify your fix works by running your script again
        5. Test edge cases to ensure your fix is robust
        6. Submit your changes and finish your work by issuing the following command: `echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT`.
           Do not combine it with any other command. <important>After this command, you cannot continue working on this task.</important>

        ## Important Rules

        1. Every response must contain exactly one action
        2. The action must be enclosed in <action> tags
        3. Directory or environment variable changes are not persistent. Every action is executed in a new subshell.
           However, you can prefix any action with `MY_ENV_VAR=MY_VALUE cd /path/to/working/dir && ...` or write/load environment variables from files

        <system_information>
        {{system}} {{release}} {{version}} {{machine}}
        </system_information>

        ## Formatting your response

        Here is an example of a correct response:

        <example_response>
        THOUGHT: I need to understand the structure of the repository first. Let me check what files are in the current directory to get a better understanding of the codebase.

        <action>ls -la</action>
        </example_response>

        ## Useful command examples

        ### Create a new file:

        <action>cat <<'EOF' > newfile.py
        import numpy as np
        hello = "world"
        print(hello)
        EOF</action>

        ### View file content:

        <action>nl -ba filename.py | sed -n '10,20p'</action>

      format_error_template: |
        Please always provide EXACTLY ONE action in <action> tags, found {{actions|length}} actions.
        If you want to end the task, please issue the following command: `echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT`
        without any other command.
        Else, please format your response exactly as follows:

        <response_example>
        Here are some thoughts about why you want to perform the action.

        <action>your_command</action>
        </response_example>

      action_observation_template: |
        <returncode>{{output.returncode}}</returncode>
        {% if output.output | length < 10000 -%}
        <output>
        {{ output.output -}}
        </output>
        {%- else -%}
        <warning>
        The output of your last command was too long.
        Please try a different command that produces less output.
        </warning>
        {%- set elided_chars = output.output | length - 10000 -%}
        <output_head>
        {{ output.output[:5000] }}
        </output_head>
        <elided_chars>
        {{ elided_chars }} characters elided
        </elided_chars>
        <output_tail>
        {{ output.output[-5000:] }}
        </output_tail>
        {%- endif -%}

      step_limit: 0
      cost_limit: 3.0
    ```

??? example "Default markdown format"

    This is the default configuration (already built-in, you don't need to specify this):

    ```yaml
    agent:
      action_regex: "```bash\\s*\\n(.*?)\\n```"
      system_template: |
        Your response must contain exactly ONE bash code block.

        ```bash
        your_command_here
        ```
    ```


## Model configuration

See [this guide](../models/quickstart.md) for more details on model configuration.

## Environment configuration

See [this guide](../advanced/environments.md) for more details on environment configuration.

## Run configuration

See the information in "Usage".

{% include-markdown "_footer.md" %}

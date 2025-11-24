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

!!! warning "Important"

    If you set a custom action_regex (e.g. `<action>(.*?)</action>`), you must use the same output format across all prompt templates (system_template, instance_template, format_error_template, etc.), ensuring the LLM wraps commands accordingly. See the example below for a complete configuration.


??? example "Using XML format instead of markdown"

    This example uses the same structure as the default mini.yaml config, but with `<action>` tags instead of markdown code blocks:

    ```yaml
    --8<-- "src/minisweagent/config/extra/swebench_xml.yaml"
    ```

    You can also directly load this config by specifying `--config swebench_xml`.


??? example "Default markdown format"

    This is the default configuration (already the default, you don't need to specify this):


    ```yaml
    agent:
      action_regex: ```bash\s*\n(.*?)\n```
      system_template: |
        Your response must contain exactly ONE bash code block.

        ```bash
        your_command_here
        ```
    ```

!!! warning "Linebreaks & escaping"

    When specifying `action_regex` from the `yaml` config file, make sure you understand how escaping in yaml files works.
    For example, when you use the `|` primitive, your regex might have a linbreak at the end which is probably not what you want.
    The best way is to keep your regex on a single line and NOT use any quotation marks around it. You do NOT need to escape any characters in the regex. Example: `action_regex: <bash_code>(.*?)</bash_code>`

## Model configuration

See [this guide](../models/quickstart.md) for more details on model configuration.

## Environment configuration

See [this guide](../advanced/environments.md) for more details on environment configuration.

## Run configuration

See the information in "Usage".

{% include-markdown "_footer.md" %}

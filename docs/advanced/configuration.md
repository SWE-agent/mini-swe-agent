# Configuration

## Environment variables and global configuration

All global configuration can be either set as environment variables, or in the `.env` file (the exact location is printed when you run `micro`).

!!! note "Precedence"

    Environment variables take precedence over variables set in the `.env` file.

We provide several helper functions to update the global configuration.

For example, to set the default model and API keys, you can run:

```bash
micro-extra config setup
```

or to update specific settings:

```
micro-extra config set KEY VALUE
# e.g.,
micro-extra config set MSWEA_MODEL_NAME "claude-sonnet-4-20250514"
micro-extra config set MSWEA_MODEL_API_KEY "sk-..."
```

or to unset a key:

```bash
micro-extra config unset KEY
# e.g.,
micro-extra config unset MSWEA_MODEL_API_KEY
```

You can also edit the `.env` file directly and we provide a helper function for that:

```bash
micro-extra config edit
```

### Models and keys

```bash
# Default model name
# (default: not set)
MSWEA_MODEL_NAME="claude-sonnet-4-20250514"

# Default API key
# (default: not set)
MSWEA_MODEL_API_KEY="sk-..."
```

For Anthropic models, you can also use `ANTHROPIC_API_KEYS` for advanced parallel execution:

```bash
# Multiple Anthropic keys for parallel execution (separated by "::")
ANTHROPIC_API_KEYS="key1::key2::key3"
```

This allows different threads to use different API keys to avoid prompt caching conflicts when running multiple agents in parallel.

Global cost limits:

```bash
# Global limit on number of model calls (0 = no limit)
# (default: 0)
MSWEA_GLOBAL_CALL_LIMIT="100"

# Global cost limit in dollars (0 = no limit)
# (default: 0)
MSWEA_GLOBAL_COST_LIMIT="10.00"
```

### Default config files

```bash
# Set a custom directory for agent config files in addition to the builtin ones
# This allows to specify them by names
MSWEA_CONFIG_DIR="/path/to/your/own/config/dir"

# Config path for GitHub issue script
# (default: package_dir / "config" / "github_issue.yaml")
MSWEA_GITHUB_CONFIG_PATH="/path/to/your/github/config.yaml"

# Custom style path for trajectory inspector
# (default: package_dir / "config" / "local2.tcss")
MSWEA_INSPECTOR_STYLE_PATH="/path/to/your/inspector/style.tcss"

# Config path for local2 textual interface script
# (default: package_dir / "config" / "local.yaml")
MSWEA_LOCAL2_CONFIG_PATH="/path/to/your/local2/config.yaml"

# Custom style path for local2 textual interface
# (default: package_dir / "config" / "local2.tcss")
MSWEA_LOCAL2_STYLE_PATH="/path/to/your/local2/style.tcss"

# Config path for local run script
# (default: package_dir / "config" / "local.yaml")
MSWEA_LOCAL_CONFIG_PATH="/path/to/your/own/config"
```

## Default run files

```bash
# Default run script entry point for the main CLI
# (default: "microsweagent.run.micro")
MSWEA_DEFAULT_RUN="microsweagent.run.micro"
```

## Agent configuration files

Configuration files look like this:

??? note "Configuration file"

    ```yaml
    --8<-- "src/microsweagent/config/local.yaml"
    ```

We use [Jinja2](https://jinja.palletsprojects.com/) to render templates (e.g., the instance template).
TL;DR: You include variables with double curly braces, e.g. `{{task}}`, but you can also do fairly complicated logic like this:

??? note "Example: Dealing with long observations"

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

- Environment variables
- Agent config variables
- Environment config variables
- Explicitly passed variables (`observation`, `task` etc.) depending on the template

{% include-markdown "_footer.md" %}
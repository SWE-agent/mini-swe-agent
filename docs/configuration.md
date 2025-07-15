# Configuration

## Environment variables

All global configuration can be either set as environment variables, or in the `.env` file (the exact location is printed when you run `micro`).

!!! note "Precedence"

    Environment variables take precedence over variables set in the `.env` file.

```bash
# Default run script entry point for the main CLI
# (default: "microsweagent.run.local")
MSWEA_DEFAULT_RUN="microsweagent.run.local"

# Config path for GitHub issue script
# (default: package_dir / "config" / "github_issue.yaml")
MSWEA_GITHUB_CONFIG_PATH="/path/to/your/github/config.yaml"

# Global limit on number of model calls (0 = no limit)
# (default: 0)
MSWEA_GLOBAL_CALL_LIMIT="100"

# Global cost limit in dollars (0 = no limit)
# (default: 0)
MSWEA_GLOBAL_COST_LIMIT="10.00"

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

# Default model name
# (default: not set)
MSWEA_MODEL_NAME="claude-sonnet-4-20250514"

# Default API key
# (default: not set)
MSWEA_MODEL_API_KEY="sk-..."
```

### Anthropic Models

For Anthropic models, you can also use `ANTHROPIC_API_KEYS` for advanced parallel execution:

```bash
# Multiple Anthropic keys for parallel execution (separated by "::")
ANTHROPIC_API_KEYS="key1::key2::key3"
```

This allows different threads to use different API keys to avoid prompt caching conflicts when running multiple agents in parallel.

## Configuration files

Configuration files look like this:

??? note "Configuration file"

    ```yaml
    --8<-- "src/microsweagent/config/local.yaml"
    ```

We use [Jinja2](https://jinja.palletsprojects.com/) to render templates (e.g., the instance template).
TL;DR: You include variables with double curly braces, e.g. `{{task}}`, but you can also do fairly complicated logic like this:

??? note "Dealing with long observations"

    ```jinja
    {%- set full_output -%}
        <returncode>
            {{output.returncode}}
        </returncode>
        <stderr>
            {{output.stderr}}
        </stderr>
        <stdout>
            {{output.stdout}}
        </stdout>
    {%- endset -%}

    {%- if full_output | length < 10000 -%}
        {{ full_output }}
    {%- else -%}
        <warning>
            The output of your last command was too long.
        </warning>

        {%- set elided_chars = full_output | length - 10000 -%}

        <observation_head>
            {{ full_output[:5000] }}
        </observation_head>

        <elided_chars>
            {{ elided_chars }} characters elided
        </elided_chars>

        <observation_tail>
            {{ full_output[-5000:] }}
        </observation_tail>
    {%- endif -%}
    ```

{% include-markdown "_footer.md" %}
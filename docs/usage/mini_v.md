# `mini -v`

!!! abstract "Overview"

    * `mini -v` is a pager-style interactive command line interface for using mini-SWE-agent in the local requirement (as opposed for workflows that require sandboxing or large scale batch processing).
    * Compared to [`mini`](mini.md), `mini -v` offers a more advanced UI based on [Textual](https://textual.textualize.io/).

!!! tip "Feedback wanted!"
    Give feedback on the `mini` and `mini -v` interfaces at [this github issue](https://github.com/swe-agent/mini-swe-agent/issues/161)
    or in our [Slack channel](https://join.slack.com/t/swe-bench/shared_invite/zt-36pj9bu5s-o3_yXPZbaH2wVnxnss1EkQ).


<figure markdown="span">
  <div class="gif-container gif-container-styled" data-glightbox-disabled>
    <img src="https://github.com/SWE-agent/swe-agent-media/blob/main/media/mini/png/mini2.png?raw=true"
         data-gif="https://github.com/SWE-agent/swe-agent-media/blob/main/media/mini/gif/mini2.gif?raw=true"
         alt="miniv" data-glightbox="false" width="600" />
  </div>
</figure>

## Command line options

Invocation

```bash
mini -v [other options]
```

!!! tip "Default visual mode"

    If you want to use the visual mode by default, you can set the `MSWEA_VISUAL_MODE_DEFAULT` environment variable to `true`
    (`mini-extra config set MSWEA_VISUAL_MODE_DEFAULT true`).

Useful switches:

- `-h`/`--help`: Show help
- `-t`/`--task`: Specify a task to run (else you will be prompted)
- `-c`/`--config`: Specify a config file to use, else we will use [`mini.yaml`](https://github.com/swe-agent/mini-swe-agent/blob/main/src/minisweagent/config/mini.yaml) or the config `MSWEA_MINI_CONFIG_PATH` environment variable (see [global configuration](../advanced/global_configuration.md))
  It's enough to specify the name of the config file, e.g., `-c mini.yaml` (see [global configuration](../advanced/global_configuration.md) for how it is resolved).
- `-m`/`--model`: Specify a model to use, else we will use the model `MSWEA_MODEL_NAME` environment variable (see [global configuration](../advanced/global_configuration.md))
- `-y`/`--yolo`: Start in `yolo` mode (see below)

## Key bindings

!!! tip "Focused input fields"

    Whenever you are prompted to enter text, the input field will be focused.
    You can use `Tab` or `Esc` to switch between the input field controls and the general controls below.

- `f1` or `?`: Show keybinding help
- `q` (or `ctrl+q`): Quit the agent
- `c`: Switch to `confirm` mode
- `y` (or `ctrl+y`): Switch to `yolo` mode
- `h` or `LEFT`: Go to previous step of the agent
- `l` or `RIGHT`: Go to next step of the agent
- `0`: Go to first step of the agent
- `$`: Go to last step of the agent
- `j` or `DOWN`: Scroll down
- `k` or `UP`: Scroll up

## Modes of operation

`mini -v` provides two different modes of operation

- `confirm` (`c`): The LM proposes an action and the user is prompted to confirm (press Enter)) or reject (enter a rejection message))
- `yolo` (`y`): The action from the LM is executed immediately without confirmation
- `human` (`u`): The user is prompted to enter a command directly

You can switch between the modes at any time by pressing the `c`, `y`, or `u` keys.

`mini -v` starts in `confirm` mode by default. To start in `yolo` mode, you can add `-y`/`--yolo` to the command line.

### FAQ

> How can I select/copy text on the screen?

Hold down the `Alt`/`Option` key and use the mouse to select the text.

## Miscellaneous tips

- `mini` saves the full history of your last run to your global config directory.
  The path to the directory is printed when you start `mini`.

## Implementation

??? note "Default config"

    - [Read on GitHub](https://github.com/swe-agent/mini-swe-agent/blob/main/src/minisweagent/config/mini.yaml)

    ```yaml
    --8<-- "src/minisweagent/config/mini.yaml"
    ```

??? note "Run script"

    - [Read on GitHub](https://github.com/swe-agent/mini-swe-agent/blob/main/src/minisweagent/run/mini.py)
    - [API reference](../reference/run/mini.md)

    ```python
    --8<-- "src/minisweagent/run/mini.py"
    ```

??? note "Agent class"

    - [Read on GitHub](https://github.com/swe-agent/mini-swe-agent/blob/main/src/minisweagent/agents/interactive.py)
    - [API reference](../reference/agents/interactive.md)

    ```python
    --8<-- "src/minisweagent/agents/interactive.py"
    ```

{% include-markdown "../_footer.md" %}

# Inspector: Browse agent trajectories

!!! abstract "Overview"

    * The `inspector` is a tool that allows you to browse `.traj.json` files that show the history of a micro-SWE-agent run.
    * Quickly start it with `micro-e i` or `micro-extra inspector`.

## Usage

```bash
# Find all .traj.json files recursively from current directory
micro-extra inspector
# or shorter
micro-e i
# Open the inspector for a specific file
micro-e i <path_to_traj.json>
# Search for trajectory files in a specific directory
micro-e i <path_to_directory>
```

## Key bindings

- `q`: Quit the inspector
- `h`/`LEFT`: Previous step
- `l`/`RIGHT`: Next step
- `j`/`DOWN`: Scroll down
- `k`/`UP`: Scroll up
- `H`: Previous trajectory
- `L`: Next trajectory

### FAQ

> How can I select/copy text on the screen?

Hold down the `Alt`/`Option` key and use the mouse to select the text.

## Implementation

The inspector is implemented with [textual](https://textual.textualize.io/).

??? note "Implementation"

    - [Read on GitHub](https://github.com/swe-agent/micro-swe-agent/blob/main/src/microsweagent//run/inspector.py)

    ```python linenums="1"
    --8<-- "src/microsweagent/run/inspector.py"
    ```

{% include-markdown "../_footer.md" %}
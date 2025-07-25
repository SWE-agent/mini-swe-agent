# Inspector: Browse agent trajectories

!!! abstract "Overview"

    * The `inspector` is a tool that allows you to browse `.traj.json` files that show the history of a mini-SWE-agent run.
    * Quickly start it with `mini-e i` or `mini-extra inspector`.

<figure markdown="span">
  <div class="gif-container gif-container-styled" data-glightbox-disabled>
    <img src="https://github.com/SWE-agent/swe-agent-media/blob/main/media/mini/png/inspector.png?raw=true"
         data-gif="https://github.com/SWE-agent/swe-agent-media/blob/main/media/mini/gif/inspector.gif?raw=true"
         alt="inspector" data-glightbox="false" width="600" />
  </div>
</figure>

## Usage

```bash
# Find all .traj.json files recursively from current directory
mini-extra inspector
# or shorter
mini-e i
# Open the inspector for a specific file
mini-e i <path_to_traj.json>
# Search for trajectory files in a specific directory
mini-e i <path_to_directory>
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

    - [Read on GitHub](https://github.com/swe-agent/mini-swe-agent/blob/main/src/minisweagent//run/inspector.py)

    ```python linenums="1"
    --8<-- "src/minisweagent/run/inspector.py"
    ```

{% include-markdown "../_footer.md" %}
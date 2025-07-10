# Quick start

!!! tip "Installation Options"

    === "pipx"

        Use pipx to install & run `micro` in an isolated environment.

        First [install pipx](https://pipx.pypa.io/stable/installation/), then

        ```bash
        # Simple UI
        pipx run --spec git+ssh://git@github.com/SWE-agent/micro-swe-agent micro
        # Textual UI
        pipx run --spec git+ssh://git@github.com/SWE-agent/micro-swe-agent micro2
        ```

        If the invocation doesn't immediately work, you might need to run `pipx ensurepath`.

    === "pip"

        Use pip to install `micro` in your current environment:

        ```bash
        pip install micro-swe-agent
        ```

        Then run:

        ```bash
        # Simple UI
        micro
        # Textual UI
        micro2
        ```


    === "From source"

        For development or if you want to customize the agent:

        ```bash
        git clone https://github.com/SWE-agent/micro-swe-agent.git
        cd micro-swe-agent
        pip install -e .
        ```

        Then run:

        ```bash
        # Simple UI
        micro
        # Textual UI
        micro2
        ```

        Or pick a [run script](reference/run/):

        ```bash
        python microsweagent/run/hello_world.py
        ```

??? example "Example Prompts"

    Try micro-SWE-agent with these example prompts:

    - Implement a Sudoku solver in python in the `sudoku` folder. Make sure the codebase is modular and well tested with pytest.
    - Please run pytest on the current project, discover failing unittests and help me fix them. Always make sure to test the final solution.
    - Help me document & type my codebase by adding short docstrings and type hints.

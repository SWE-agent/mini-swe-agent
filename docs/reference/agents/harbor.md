# HarborMiniAgent

!!! note "HarborMiniAgent class"

    - [Read on GitHub](https://github.com/swe-agent/mini-swe-agent/blob/main/src/minisweagent/agents/harbor.py)

    ??? note "Full source code"

        ```python
        --8<-- "src/minisweagent/agents/harbor.py"
        ```

!!! tip "Usage"

    Use this agent for incremental trajectory saving.

    ```bash
    mini --agent-class harbor -t "your task" -o trajectory.json
    ```

::: minisweagent.agents.harbor.HarborMiniAgent

{% include-markdown "../../_footer.md" %}

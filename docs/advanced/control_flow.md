# Agent control flow

!!! abstract "Understanding the default agent"

    * This guide shows the control flow of the default agent.
    * After this, you're ready to [remix & extend mini](cookbook.md)

The following diagram shows the control flow of the mini agent:

<div align="center">
    <img src="../../assets/mini_control_flow.svg" alt="Agent control flow" style="max-width: 600px;" />
</div>

And here is the code that implements it:

??? note "Default agent class"

    - [Read on GitHub](https://github.com/swe-agent/mini-swe-agent/blob/main/src/minisweagent/agents/default.py)
    - [API reference](../reference/agents/default.md)

    ```python
    --8<-- "src/minisweagent/agents/default.py"
    ```

Essentially, `DefaultAgent.run` calls `DefaultAgent.step` in a loop until the agent has finished its task.

The `step` method is the core of the agent. It does the following:

1. Queries the model for a response based on the current messages (`DefaultAgent.query`, calling `Model.query`)
2. Parses the response to get the action, i.e., the shell command to execute (`DefaultAgent.parse_action`)
3. Executes the action in the environment (`DefaultAgent.execute_action`, calling `Environment.execute`)
4. Renders the observation message with `DefaultAgent.render_template`
5. Adds the observation to the messages

The interesting bit is how we handle error conditions and the finish condition:
This uses exceptions of two types: `TerminatingException` and `NonTerminatingException`.

- `TerminatingException` is raised when the agent has finished its task or we hit a limit (cost, step limit, etc.)
- `NonTerminatingException` is raised when the agent has not finished its task, but we want to continue the loop.
   In this case, all we need to do is to add a new message to the messages list, so that the LM can see the new state.
   There are two typical cases that we handle this way:

    1. `TimeoutError`: the action took too long to execute (we show partial output)
    2. `FormatError`: the output from the LM contained zero or multiple actions (we show the error message)

The `DefaultAgent.run` method catches these exceptions and handles them by adding the corresponding message to the messages list and continuing the loop.

```python
while True:
    try:
        self.step()
    except NonTerminatingException as e:
        self.add_message("user", str(e))
    except TerminatingException as e:
        self.add_message("user", str(e))
        return type(e).__name__, str(e)
```

Using exceptions for the control flow is a lot easier than passing around flags and states, especially when extending or subclassing the agent.

{% include-markdown "_footer.md" %}
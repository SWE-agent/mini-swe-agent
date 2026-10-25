## This step: log the tool executions

The logger is already copied to `{logger}`; import it where you use it, the way this code imports its own modules.

A checker found these tool calls: they run between an LLM reply and the next LLM call.
{sites}

Find the one function they all go through (follow their callers up to a registry, dispatcher or `execute` method) and log there, once for every tool:

```python
start = observatory_log.tool_start(name, args)
try:
    out = self.tools[name](**args)
except Exception as error:
    observatory_log.tool_error(start, name, error)
    raise
observatory_log.tool_end(start, name, result=out)
```

- `result` is the whole output, the variable itself: not a slice, one field or a count.
- If the agent ends a task by raising an exception (for example on submit), report it with `tool_end` and a short result such as "submitted", not with `tool_error`, and re-raise it.
- Keep the code's own structure: wrap the existing call, do not move it into a new loop or function.

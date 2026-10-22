## This step: log the LLM calls

The logger is already copied to `{logger}`; import it where you use it, the way this code imports its own modules. Its docstring lists the six functions.

The calls that return the model's reply:
{sites}

Log each of them, in the function that contains it (if one wrapper function contains all of them, log there once):
- `observatory_log.llm_query(step, model=..., messages=messages)` right before the call.
- `observatory_log.llm_response(step, actions=[...], reply=reply, response=response)` right after the reply was parsed (`actions`: the names of the actions or tool calls it asks for, `[]` if none).
- If the call raises: `observatory_log.llm_error(step, error)` and re-raise with a bare `raise`, so that the agent's own error handling runs as before.

Log whole values: a checker verifies that the model reply reaches a logger call as the variable itself. Pass the variable (`response=response`, `reply=reply`, `messages=messages`), not a count, a slice, one field or a value built from it; such summaries may be added as extra fields. `step` is the agent's step counter where the function has one, else `None`. For example:

```python
observatory_log.llm_query(step, model=self.model, messages=messages)
try:
    response = self.client.messages.create(...)  # the existing call, unchanged
except Exception as error:
    observatory_log.llm_error(step, error)
    raise
reply = self.parse(response)
observatory_log.llm_response(step, actions=[c["name"] for c in reply["calls"]], reply=reply, response=response)
```

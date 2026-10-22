## This step: log the user input

The logger is already copied to `{logger}`; import it where you use it, the way this code imports its own modules.

Where the user's text comes in:
{sites}

Report each one as the tool action `"user_input"`:
- The task, at the start of the run function, before it is used:
  ```python
  start = observatory_log.tool_start("user_input", {"task": task})
  observatory_log.tool_end(start, "user_input", result=task)
  ```
- Each call that reads what the user types: `tool_start` with the prompt right before it, `tool_end` with the variable the call returned right after it:
  ```python
  start = observatory_log.tool_start("user_input", {"prompt": question})
  answer = input(question)
  observatory_log.tool_end(start, "user_input", result=answer)
  ```

## This step: log the user input

The logger is already copied to `{logger}`; import it where you use it, the way this code imports its own modules.

Where the user's text comes in:
{sites}

Report each one with a single call, `observatory_log.user_input(answer, prompt=...)`, right after the text is there, passing the variable that holds it:
- The task: at the start of the run function, before it is used:
  ```python
  observatory_log.user_input(task)
  ```
- Each call that reads what the user types: one new line right after it. The line that reads the input stays exactly as it is:
  ```python
  <the existing line, unchanged, for example: answer = input(f"{question} ")>
  observatory_log.user_input(answer, prompt=question)
  ```

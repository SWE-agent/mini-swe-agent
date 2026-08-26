# mini-SWE-agent overview

- mini-SWE-agent implements an AI software engineering agent that solves github issues and similar programming challenges
- The idea of this project is to write the simplest, smallest, most readable agent.

The project is structured as

```bash
minisweagent/__init__  # Protocols/interfaces for all base classes
minisweagent/agents  # Agent control flow & loop
minisweagent/environments  # Executing agent actions
minisweagent/models  # LM interfaces
minisweagent/run  # Run scripts that serve as an entry point
```

- The project embraces polymorphism: Every individual class should be simple, but we offer alternatives
- Every use case should start with a run script, that picks one agent, environment, and model class to run

# Style guide

1. Target python 3.10 or higher
2. Use python with type annotations. Use `list` instead of `List`.
3. Use `pathlib` instead of `os.path`. Use `Path.read_text()` over `with ...open()` constructs.
4. Use `typer` to add interfaces
5. Keep code comments to a minimum and only highlight particularly logically challenging things
6. Do not append to the README unless specifically requested
7. Use `jinja` for formatting templates
8. Use `dataclass` for keeping track config
9. Do not catch exceptions unless explicitly told to.
10. Write concise, short, minimal code.
11. In most cases, avoid initializing variables just to pass them to a function. Instead just pass the expression to the function directly.
12. Not every exception has to be caught. Exceptions are a good way to show problems to a user.
13. This repository rewards minimal code. Try to be as concise as possible.

Here's an example for rule 11:

```python
# bad
a = func()
Class(a)

# good
Class(func())
```

## Test style

1. Use `pytest`, not `unittest`.
2. <IMPORTANT>Do not mock/patch anything that you're not explicitly asked to do</IMPORTANT>
3. Avoid writing trivial tests. Every test should test for at least one, preferably multiple points of failure
4. Avoid splitting up code in multiple lines like this: `a=func()\n assert a=b`. Instead, just do `assert func() == b`
5. The first argument to `pytest.mark.parametrize` should be a tuple (not a string! not a list!), the second argument must be a list (not a tuple!).

Here's an example for rule 4:

```python
# bad
result = func()
assert result == b

# good
assert func() == b
```

# Commit messages

Use the following format for commit messages:

- `ci: description` for all testing related changes, and changes to github workflows etc.
- `dev: description` for development related changes, including updates to the cursor or claude rules
- `fix(component): description` for bug fixes
- `feat(component): description` for new features
- `enh(component): description` for enhancements
- `docs: description` for documentation
- `ref(component): description` for refactoring
- `chore: description` for maintenance tasks (pre-commit hooks, imports, etc.)

Generally, the description should focus on the intent of the changes, not the implementation details.

## Style notes

<IMPORTANT>Do **NOT** add "Co-authored-by: Cursor" lines to the commit message or to the trailer.</IMPORTANT>

## Reviewing

While preparing the commit message, flag critical issues that should be addressed before committing. Do not flag style issues or minor changes.

Flag the following:

- Anything that might raise an unhandled exception in an unintentional manner
- Anything that looks logically wrong or inconsistent
- Breaking changes to protocols/interfaces without corresponding updates to implementations

Flag the following style issue as minor:

- Imports not at top of file

## Components

Use these component names in parentheses for `fix`, `feat`, `enh`, and `ref` commits:

- `models` - Changes to model interfaces (litellm, anthropic, openai, portkey, openrouter)
- `agents` - Changes to agent classes (default, interactive, multimodal)
- `env` - Changes to environments (docker, local, singularity, bubblewrap, swerex)
- `config` - Changes to configuration files or config handling
- `run` - Changes to run scripts (mini, hello_world)
- `benchmarks` - Changes to benchmark runners (swebench, inspector)
- `cli` - Changes to CLI argument handling
- `deps` - Dependency updates

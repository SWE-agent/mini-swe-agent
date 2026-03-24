# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Mini-SWE-Agent is a minimal AI software engineering agent (~100-line agent class) that achieves >74% on SWE-bench verified. It uses three protocol-based abstractions: **Model**, **Environment**, and **Agent**, composed via YAML configs.

## Commands

### Install
```bash
uv pip install -e '.[dev]'       # development
uv pip install -e '.[full]'      # all extras (modal, swe-rex, etc.)
```

### Lint & Format
```bash
ruff check --fix && ruff format  # lint and format
pre-commit run --all-files       # all pre-commit hooks
pylint minisweagent/ --errors-only
```

### Test
```bash
pytest -v --cov --cov-branch -n auto         # all tests (parallel)
pytest tests/agents/test_default.py          # single file
pytest -v --run-fire                         # include tests with real API calls
```

### Run
```bash
mini                                          # interactive agent (prompts for task)
mini --config src/minisweagent/config/mini.yaml --task "fix the bug"
```

## Architecture

Three protocols defined in `src/minisweagent/__init__.py`:
- **Model**: wraps an LLM (query, format_message, format_observation_messages)
- **Environment**: executes shell actions (execute, get_template_vars)
- **Agent**: orchestrates model + environment (run, save)

### Agent Loop (`src/minisweagent/agents/default.py`)
1. Render `system_template` → add as system message
2. Render `instance_template` with task → add as user message
3. Loop: `query()` → parse bash actions → `execute()` in environment → format observations → append to messages
4. On exit message: save trajectory JSON

### Key Directories
- `src/minisweagent/agents/` — `default.py` (core loop, ~156 lines), `interactive.py` (human-in-the-loop)
- `src/minisweagent/models/` — `litellm_model.py` (default, supports all providers), plus openrouter/portkey/requesty variants
- `src/minisweagent/environments/` — `local.py` (subprocess), `docker.py`, `singularity.py`, plus extras (bubblewrap, modal, swe-rex)
- `src/minisweagent/config/` — YAML configs (`mini.yaml` is the default); Jinja2 templates for system/instance/observation prompts
- `src/minisweagent/run/` — CLI entry points (`mini.py` is the main one, typer-based)
- `src/minisweagent/utils/` — action parsers (`actions_toolcall.py`, `actions_text.py`), retry logic, cache control

### Config System
YAML files specify model class, environment class, agent class, and their configs. CLI args override YAML values. Jinja2 templates (with `StrictUndefined`) are used for all prompts.

## Code Style (from `.github/copilot-instructions.md`)

- Python 3.10+, type annotations using built-ins (`list` not `List`)
- `pathlib` over `os.path`
- `typer` for CLI, `dataclasses` for config, `jinja2` for templates
- Minimal comments — only for complex logic
- Don't catch exceptions unless explicitly needed
- Prefer concise, minimal code

### Test Style
- pytest (not unittest); no mocking/patching unless asked
- Test multiple failure points per test; keep assertions on one line
- `parametrize`: first arg is a tuple, second arg is a list
- `--run-fire` flag gates tests that make real API calls

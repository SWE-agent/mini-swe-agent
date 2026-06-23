# v2.0 Migration Guide

!!! danger "Breaking Changes"

    **mini-swe-agent v2.0** brings major improvements but requires migration.
    To stay with v1.x, pin your dependency: `mini-swe-agent~=1.0`.
    See the [v1 documentation](https://mini-swe-agent.com/v1/) or the [v1 branch on GitHub](https://github.com/SWE-agent/mini-swe-agent/tree/v1).

## What's new

- **Tool calls**: Native tool calling API support (now the default)
- **Multimodal input**: Support for images and other content types

## What do I need to change?

> I only use the mini CLI with default configs

No changes needed.

> I use custom configs

You might need to move some config keys. See [Config changes](#config-changes).

> I parse/analyze trajectories

Some metadata fields moved to `extra`. See [Trajectory format](#trajectory-format).

> I use the python bindings or built custom subclasses

You will need to refactor. An agent can refactor your code based on the instructions below.

## Config changes

If you only changed `system_template` and `instance_template`, no changes needed.

**Move from `agent` to `model`:**

- `observation_template` (renamed from `action_observation_template`)
- `format_error_template`
- `action_regex` (only for text-based parsing)

**Removed:**

- `timeout_template`

**Code block format changed:**

Default action regex changed from `` ```bash `` to `` ```mswea_bash_command `` to avoid conflicts with bash examples in prompts.

**Completion signal changed:**

From `echo MINI_SWE_AGENT_FINAL_OUTPUT` to `echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT`.

**New structure:**

```yaml
agent:
  system_template: "..."
  instance_template: "..."
  step_limit: 0
  cost_limit: 3.
environment:
  cwd: "..."
  timeout: 30
model:
  model_name: "..."
  observation_template: "..."
  format_error_template: "..."
```

**CLI now supports multiple configs and key-value overrides:**

```bash
mini -c mini.yaml -c model.model_kwargs.temperature=0.5
mini -c swebench.yaml -c agent.step_limit=100
mini -c mini.yaml -c /path/to/model.yaml
```

!!! warning
    If you use `-c`, you will not load the full default config, so make sure to always specify your config file in addition to any overrides/updates.

## Tool calling

v2.0 uses native tool calling by default (instead of regex-based text parsing).

**What's the difference?**

- Tool calling (default): Model uses native tool calling API to invoke a "bash" tool
- Text-based (legacy): Model outputs commands in markdown code blocks (or similar), regex extracts them

**How to use it:**

Tool calling is the default. The CLI uses `mini.yaml` and `swebench.yaml` which are configured for tool calling.

```bash
# Default (tool calling)
mini
python -m minisweagent.run.benchmarks.swebench
mini-extra swebench

# Text-based parsing
mini-extra swebench -c swebench_backticks.yaml
mini -c mini_textbased.yaml
```

**For custom configs:**

```yaml
model:
  model_class: litellm  # tool calling (default)
  model_name: anthropic/claude-sonnet-4-5-20250929

# or for text-based:
model:
  model_class: litellm_textbased
  action_regex: "```mswea_bash_command\\s*\\n(.*?)\\n```"
```

## Trajectory format

- In v1, all messages in the trajectory's `messages` field were "hand-formatted", i.e., had a `content: str` and `role: str` field.
- **In v2, we use the model's native output as the basis for the message and only add the `extra` field to it**.

If you use a model that uses the standard `/completion` endpoint, then you will still always have a `content: str` and `role: str` field.
However, if you use the [`/response`](https://platform.openai.com/docs/api-reference/responses) endpoint, then things might look different.

In other words: the exact message structure depends on the model you use.

* Advantage: Any model message structure is supported
* Disadvantage: If you parse trajectories, you might need to adapt to the message structure of the model you use.

## Removed & renamed

**Removed features:**

- **"Visual" UI**: The `-v` flag for the alternate, textual based `mini -v` CLI is no longer supported. This was a tough decision to make, but in the end the visual mode didn't see the adoption we wanted and is significantly more complex to maintain than the default interface.
- **Rotating API keys**: `ANTHROPIC_API_KEYS` with `::` separator no longer supported. Use single `ANTHROPIC_API_KEY`.
- **`github_issue` run script**: The dedicated `github_issue.py` run script was removed. Use the `mini` CLI instead.
- **`MSWEA_MODEL_API_KEY` environment variable**: No longer used to override API keys.

**Removed model classes:**

- **`anthropic` model class**: Removed. Use `litellm` model class for Anthropic models (make sure that cache control is enabled).

**Renamed model classes:**

| v1 name | v2 name |
|---------|---------|
| `litellm_response_api` | `litellm_response` |
| `portkey_response_api` | `portkey_response` |

**New model classes:**

| Name | Description |
|------|-------------|
| `litellm_textbased` | Text-based parsing (regex) instead of tool calls |
| `openrouter_textbased` | Text-based parsing for OpenRouter |
| `openrouter_response` | OpenRouter with response API |

**New environment:**

- **`swerex_modal`**: Run environments on Modal (requires `pip install mini-swe-agent[modal]`)

## Architecture changes

1. **Responsibility shift**: Models now parse actions and format observations. This enables switching between tool calls and text parsing by changing model classes. The Agent class is now a simpler coordinator.

2. **Stateless models**: Cost tracking moved to Agent. The `cost` and `n_calls` attributes were removed from the Model protocol.

3. **Pydantic configs**: `AgentConfig` (and other configs) changed from `dataclass` to Pydantic `BaseModel`. This requires `pydantic >= 2.0`.

4. **New protocol methods**: All classes implement `get_template_vars()` and `serialize()` instead of requiring specific attributes.

### Protocol changes

If you want to write a custom Model, Environment or Agent compatible with `mini-swe-agent`, you don't need to subclass anything.
Rather, mini-swe-agent fully uses duck typing with [protocols](https://typing.python.org/en/latest/spec/protocol.html)
(tl;dr: as long as you implement the required methods, you can use any class as a Model, Environment or Agent).
Config options like `--config-class` also take full import classes, so you can put your classes wherever you want.

**Model protocol:**

```python
# Removed attributes
cost: float      # moved to Agent
n_calls: int     # moved to Agent

# New methods
def format_message(self, **kwargs) -> dict: ...
def format_observation_messages(self, message: dict, outputs: list[dict], template_vars: dict | None = None) -> list[dict]: ...
def serialize(self) -> dict: ...
```

**Environment protocol:**

```python
# Changed signature
def execute(self, action: dict, cwd: str = "") -> dict[str, Any]: ...  # was: (command: str) -> dict[str, str]

# New method
def serialize(self) -> dict: ...
```

**Agent protocol:**

```python
# Removed attributes (you don't need to implement these anymore, but you can)
model: Model
env: Environment
messages: list[dict]

# Changed return type
def run(self, task: str, **kwargs) -> dict: ...  # was: tuple[str, str]

# New method
def save(self, path: Path | None, *extra_dicts) -> dict: ...
```

### Exception changes

All flow control exceptions now inherit from `InterruptAgentFlow` and moved to `minisweagent.exceptions`:

```python
InterruptAgentFlow (base)
├── Submitted (task completed)
├── LimitsExceeded (cost/step limit)
├── FormatError (invalid model output)
└── UserInterruption (user cancelled)  # new
```

**Removed exception classes:**

- `NonTerminatingException` - use `InterruptAgentFlow` base class
- `TerminatingException` - use `InterruptAgentFlow` base class
- `ExecutionTimeoutError` - removed (no longer used)

```python
# Old
from minisweagent.agents.default import Submitted, FormatError

# New
from minisweagent.exceptions import Submitted, FormatError
```

### Agent.run() return value

```python
# Old (v1)
submission, exit_status = agent.run(task)  # tuple[str, str]

# New (v2)
result = agent.run(task)  # dict
submission = result["submission"]
exit_status = result["exit_status"]
```

The `run()` method returns the `extra` dict from the final exit message. For full trajectory data, use `agent.save(path)` or `agent.serialize()`.

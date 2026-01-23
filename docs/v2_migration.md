## What's the objective of these changes?

`mini` v2.0 will be even more flexible & performant while staying just as simple as before.

In particular:

* support for **tool calls**
* support for **multimodal input**


## What do I need to change?

> I only use the mini CLI with the default configs

no changes needed

> I use custom configs

You might need to rename/move some config keys unless you only changed the system and instance prompt. See below.

> I need to parse & analyze trajectories

Some of the extra information fields (timestamps, extra cost information, raw outputs, etc.) have moved, see below.

> I used the python bindings/built custom subclasses

You will need to refactor. Please read the sections below. We're fairly confident that an agent can refactor your code based on the instructions below.

## Did the mini CLI change?

No. If you use the CLI with the default configs, nothing needs to change.

## How do I need to update my custom `.yaml` configs?

If you only changed `system_template` and `instance_template`, no changes needed.

Otherwise, you need to move some config keys to different sections:

**Move from `agent:` to `environment:`:**
- `action_observation_template` (how to format command output)
- `timeout_template` (message when command times out)

**Move from `agent:` to `model:`:**
- `format_error_template` (message when model output format is wrong)
- `action_regex` (regex to extract command from model output)

**New config structure:**
```yaml
agent:
  system_template: "..."
  instance_template: "..."
  step_limit: 0
  cost_limit: 3.
environment:
  cwd: "..."
  action_observation_template: "..."  # moved here
  timeout_template: "..."             # moved here
model:
  model_name: "..."
  format_error_template: "..."        # moved here
  action_regex: "..."                 # moved here
```

**CLI now supports multiple configs and key-value overrides:**
```bash
# Merge multiple config files
mini -c mini.yaml -c model.model_kwargs.temperature=0.5

# Set nested config values
mini -c swebench.yaml -c agent.step_limit=100
```

## How has the trajectory (`.traj.json`) format changed?

### Message structure now includes `extra` field

- All extra metadata (costs, timestamps, raw data) is now in `extra` rather than top-level message fields
- Model output messages include `extra.actions` (list of parsed actions) and `extra.response` (raw API response)
- Environment observation messages include `extra` with raw output, returncode, timestamps

### Code block format changed
The default action regex changed from `` ```bash`` to `` ```mswea_bash_command`` to avoid conflicts with bash examples in prompts.

### Completion signal changed
Changed from `echo MINI_SWE_AGENT_FINAL_OUTPUT` to `echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT` for clarity.

### Trajectory format version
The trajectory format version changed from `mini-swe-agent-1.0` to `mini-swe-agent-1.1`.

## What are the biggest internal changes?

1. **v2.0 moves responsibilities from Agent to Model classes.** Models now parse actions and format observations. This enables supporting both tool calls and text-based parsing by switching model classes (e.g., `LitellmToolcallModel` vs `LitellmModel`). The Agent class barely needs to know what a "message" is anymore.

2. **Models are now fully stateless.** Cost tracking moved to Agent, enabling cleaner separation of concerns where models focus purely on LLM interaction.

3. **New `get_template_vars()` and `serialize()` protocol.** All classes (Agent, Model, Environment) implement these methods instead of requiring specific attributes. This enables custom architectures like multi-agent systems while standardizing data access.

4. **Multimodal support.** All models now support multimodal input via special tags in content strings that get expanded to structured content (images, etc).

5. **Tool call support.** New `LitellmToolcallModel` uses native tool calling APIs instead of regex parsing.

6. **Config merging.** The CLI can now merge multiple config files and accept key-value overrides directly.

## Tell me more about the responsibility changes

**What changed:**

- Models parse actions from LLM output and format observations (via `parse_action()` and `format_observation_messages()`)
- Environments execute actions and return formatted observation messages (via `execute_messages()` and `format_observation()`)
- Agent coordinates by calling model and environment, then appending returned messages to `self.messages`
- Cost tracking is now in Agent rather than Model

**Why:**

- Enables different action handling strategies (tool calls vs text parsing) by swapping model classes
- Makes models stateless and focused on LLM interaction only
- Agent becomes a simpler coordinator that doesn't need to know message structure details

## How can I use tool calling?

v2.0 adds native tool calling support as an alternative to text-based action parsing.

**What's the difference?**
- Text-based (default): Model outputs bash commands in markdown code blocks, regex extracts them
- Tool calling: Model uses native tool calling API to invoke a "bash" tool function

**How to use it:**
```bash
# Use tool calling with mini CLI
mini -c mini_toolcall.yaml

# Use tool calling for SWE-bench
python -m minisweagent.run.benchmarks.swebench --config swebench_toolcall.yaml
```

**For custom configs:**
```yaml
model:
  model_class: minisweagent.models.LitellmToolcallModel
  model_name: anthropic/claude-sonnet-4-5-20250929
  # ... rest of model config
```

The `LitellmToolcallModel` class handles tool calling automatically and works with any model that supports OpenAI-style tool calling (Claude, GPT-4, etc.).

## Exception hierarchy changes

All agent flow control exceptions now inherit from `InterruptAgentFlow`:
```python
InterruptAgentFlow (base class for all flow interruptions)
├── Submitted (task completed)
├── LimitsExceeded (cost/step limit reached)
├── FormatError (invalid model output format)
├── TimeoutError (execution timeout)
└── UserInterruption (user cancelled)
```

All these exceptions can carry messages that get added to the trajectory. They moved from `minisweagent.agents.default` to `minisweagent.exceptions`.

## Import changes

### Exception imports
```python
# Old
from minisweagent.agents.default import Submitted, FormatError

# New
from minisweagent.exceptions import Submitted, FormatError
```

### Run script imports
```python
# Old
from minisweagent.run.github_issue import run_github_issue
from minisweagent.run.inspector import main

# New
from minisweagent.run.extra.github_issue import run_github_issue
from minisweagent.run.utilities.inspector import main
```

## Agent.run() return value changed

```python
# Old (v1)
submission, exit_status = agent.run(task)  # Returns tuple[str, str]

# New (v2)
result = agent.run(task)  # Returns dict
submission = result["info"]["submission"]
exit_status = result["info"]["exit_status"]
# Full trajectory data available in result
```

The `run()` method now returns the complete serialized trajectory data (same as what gets saved to `.traj.json`).

## Misc changes

### Removed rotating API keys support
The `ANTHROPIC_API_KEYS` environment variable (with `::` separator for key rotation) is no longer supported. Simply use `ANTHROPIC_API_KEY` with a single key. Key rotation is no longer necessary.

### Removed files
- `src/minisweagent/models/utils/key_per_thread.py` (key rotation utility)
- `src/minisweagent/run/utils/save.py` (saving logic moved to Agent.save())
- `src/minisweagent/config/extra/swebench_roulette.yaml`

### Renamed files
- `openai_utils.py` → `openai_response_api.py`





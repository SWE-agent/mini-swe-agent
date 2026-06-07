# Output files

!!! abstract "Overview"

    mini-SWE-agent saves run results in JSON format. This page documents the structure of these output files.

## Trajectory files (`.traj.json`)

!!! warning "v2.0 format changes"

    The output format changed in v2.0 (`trajectory_format: mini-swe-agent-1.1`). See the [v2 migration guide](../advanced/v2_migration.md) for more information.

!!! tip "Viewing trajectory files"

    Use the [inspector](inspector.md) to browse trajectory files interactively.

Trajectory files contain the full history of an agent run, including all messages, configuration, and metadata.

### Structure

```json
{
  "info": {
    "model_stats": {
      "instance_cost": 0.05,  // total cost of API calls for this run
      "api_calls": 12  // number of API calls made
    },
    "config": {
      "agent": { ... },  // agent configuration
      "agent_type": "minisweagent.agents.default.DefaultAgent",
      "model": { ... },  // model configuration
      "model_type": "minisweagent.models.litellm_model.LitellmModel",
      "environment": { ... },  // environment configuration
      "environment_type": "minisweagent.environments.local.LocalEnvironment"
    },
    "mini_version": "2.0.0",  // version of mini-SWE-agent used
    "exit_status": "Submitted",  // final status (Submitted, LimitsExceeded, etc.)
    "submission": "..."  // final output/patch submitted by the agent (if any)
  },
  "messages": [  // full conversation history
    {"role": "system", "content": "..."},
    {"role": "user", "content": "..."},
    {"role": "assistant", "content": "..."},
    ...
  ],
  "trajectory_format": "mini-swe-agent-1.1"  // format version identifier
}
```


Messages follow the [OpenAI chat format](https://platform.openai.com/docs/api-reference/chat) with an additional `extra` field for mini-SWE-agent metadata. Models may add other fields to messages (e.g., `tool_calls`, `reasoning_content`).

!!! note "Toolcall models"
    When using toolcall-based models (e.g., `LitellmToolcallModel`), the roles differ slightly: assistant messages include `tool_calls` instead of content, and observation messages use `role: "tool"` with a `tool_call_id` field.

```json
// System message (agent instructions)
{"role": "system", "content": "You are a helpful assistant..."}

// User message (task description)
{"role": "user", "content": "Please solve this issue: ..."}

// Assistant message (model response with parsed actions)
{
  "role": "assistant",
  "content": "Let me check the files...\n\n```mswea_bash_command\nls -la\n```",
  "extra": {
    "actions": [{"command": "ls -la"}],  // parsed actions to execute
    "cost": 0.003,  // cost of this API call
    "timestamp": 1706000000.0,  // unix timestamp of when this message was created
    "response": { ... }  // raw API response
  }
}

// Observation message (execution result)
{
  "role": "user",
  "content": "<returncode>0</returncode>\n<output>\nfile1.py\nfile2.py\n</output>",
  "extra": {
    "returncode": 0,
    "timestamp": 1706000001.0
  }
}

// Final message (when agent submits)
{
  "role": "user",
  "content": "",
  "extra": {
    "exit_status": "Submitted",
    "submission": "diff --git a/file.py..."
  }
}
```


## `preds.json` format

The predictions file aggregates results from all instances in a format compatible with SWE-bench evaluation:

```json
{
  "owner__repo__123": {  // keyed by instance_id
    "model_name_or_path": "anthropic/claude-sonnet-4-5-20250929",  // model used
    "instance_id": "owner__repo__123",  // SWE-bench instance identifier
    "model_patch": "diff --git a/file.py b/file.py\n..."  // generated patch (unified diff)
  },
  ...
}
```

{% include-markdown "../_footer.md" %}

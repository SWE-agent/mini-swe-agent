# Todos

* [x] spawn a new agent when it `echo`es a predefined value, like `echo MINI_SWE_AGENT_FINAL_OUTPUT`
* [x] the new agent has a different and fresh context
* [x] write to the same console (I want an indicator on the UI to know which agent I'm talking to - root / subagent)
* [x] changing mode on the parent changes the child
* [x] subagent registry is properly interpolated in system templates

## Implementation Details

### Spawning Mechanism

The Manager agent detects subagent spawn requests through echo commands with a specific subagent name:

```bash
echo "MINI_SWE_AGENT_SPAWN_CHILD::subagent-name
Task description for the subagent"
```

**Important**: Generic spawning without a subagent name is NOT allowed. You must specify a subagent name from the registry.

When this trigger is detected:
1. Manager intercepts the echo output in `execute_action()`
2. Validates that a subagent name is specified (rejects if not)
3. Extracts the subagent name and task description
4. Loads the subagent's metadata and system template from `.claude/agents/`
5. Creates a new Manager instance with the subagent's specific system template and all metadata as constructor arguments
6. Runs the child agent with `child.run(task)`
7. Returns child's output as the parent's command result

### Agent Hierarchy

Agents are identified with hierarchical IDs showing their full path:
- `ROOT` - The root agent
- `ROOT::S1` - First subagent of root
- `ROOT::S1::S2` - Second subagent of ROOT::S1

This provides clear visual indication of which agent is active in the UI.

### Mode Inheritance

All agents in a hierarchy share the same mode (human/confirm/yolo):
- Mode is stored in root agent's config
- Child agents check parent's mode via property
- Mode changes propagate to root and affect all agents
- UI clearly indicates mode changes affect entire hierarchy

### Metadata Handling

All YAML frontmatter metadata from subagent files is passed to the subagent instance:

```yaml
---
name: tdd-reviewer
description: "Ask TDD Reviewer to review..."
custom_param: "custom_value"
timeout: 300
---
```

When spawning this subagent:
- `system_template` is set to the markdown content after frontmatter
- All metadata fields (except `name`) are passed as constructor arguments
- This allows subagents to have custom configuration via their YAML metadata

### Data Flow

Subagents return data using the standard termination mechanism:
```bash
echo "MINI_SWE_AGENT_FINAL_OUTPUT
Return data line 1
Return data line 2"
```

This data flows back to the parent as command output, appearing in the parent's conversation context for further processing.

### Usage Examples

**To spawn a specific subagent:**
```bash
echo "MINI_SWE_AGENT_SPAWN_CHILD::tdd-reviewer
Please review this code change for TDD compliance"
```

**Invalid (will cause error):**
```bash
echo "MINI_SWE_AGENT_SPAWN_CHILD
Complete this general task"
```

### Resource Sharing

- **Model**: Shared across all agents for unified cost tracking
- **Environment**: Shared for consistent state (working directory, etc.)
- **Messages**: Each agent has its own message history (fresh context)
- **Config**: Each agent has its own config, except mode which is inherited

### Subagent Registry

The Manager agent uses a custom `ManagerAgentConfig` dataclass that includes:
- `subagent_registry`: Automatically loaded from `.claude/agents/` directory
- `agent_id`: Unique identifier for agent instances
- `parent_agent`: Reference to parent agent (if any)

The subagent registry is available in system templates via `{{subagent_registry}}` and lists all available subagents with their descriptions. This allows the agent to see which subagents are available for delegation.
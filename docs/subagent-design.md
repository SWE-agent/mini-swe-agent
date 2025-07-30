# Todos

* spawn a new agent when it `echo`es a predefined value, like `echo MINI_SWE_AGENT_FINAL_OUTPUT`
* the new agent has a different and fresh context
* [x] write to the same console (I want an indicator on the UI to know which agent I'm talking to - root / subagent)
* [x] changing mode on the parent changes the child

## Implementation Details

### Spawning Mechanism

The Manager agent detects subagent spawn requests through echo commands:

```bash
echo "MINI_SWE_AGENT_SPAWN_CHILD
Task description for the subagent"
```

When this trigger is detected:
1. Manager intercepts the echo output in `execute_action()`
2. Extracts the task description (everything after the trigger line)
3. Creates a new Manager instance with hierarchical agent ID
4. Runs the child agent with `child.run(task)`
5. Returns child's output as the parent's command result

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

### Data Flow

Subagents return data using the standard termination mechanism:
```bash
echo "MINI_SWE_AGENT_FINAL_OUTPUT
Return data line 1
Return data line 2"
```

This data flows back to the parent as command output, appearing in the parent's conversation context for further processing.

### Resource Sharing

- **Model**: Shared across all agents for unified cost tracking
- **Environment**: Shared for consistent state (working directory, etc.)
- **Messages**: Each agent has its own message history (fresh context)
- **Config**: Each agent has its own config, except mode which is inherited
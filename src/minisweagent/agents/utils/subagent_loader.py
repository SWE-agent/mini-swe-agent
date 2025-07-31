"""Utility to load and manage subagent configurations."""

from pathlib import Path
import yaml
import re
from typing import Dict, Tuple, Optional, Any


def load_subagent_registry(agents_dir: Path = Path(".claude/agents")) -> str:
    """Load all subagent descriptions from the agents directory for the registry."""
    if not agents_dir.exists():
        return "No subagents available."
    
    registry_lines = []
    for agent_file in sorted(agents_dir.glob("*.md")):
        content = agent_file.read_text()
        # Extract YAML frontmatter
        if match := re.match(r'^---\n(.*?)\n---', content, re.DOTALL):
            metadata = yaml.safe_load(match.group(1))
            name = metadata.get('name', agent_file.stem)
            description = metadata.get('description', 'No description provided')
            registry_lines.append(f"- {name}: {description}")
    
    return "\n".join(registry_lines) if registry_lines else "No subagents available."


def load_subagent_prompts(agents_dir: Path = Path(".claude/agents")) -> Dict[str, Dict[str, Any]]:
    """Load all subagent prompts and metadata indexed by name."""
    prompts = {}
    
    if not agents_dir.exists():
        return prompts
    
    for agent_file in sorted(agents_dir.glob("*.md")):
        content = agent_file.read_text()
        
        # Extract YAML frontmatter and content
        if match := re.match(r'^---\n(.*?)\n---\n(.*)', content, re.DOTALL):
            metadata = yaml.safe_load(match.group(1))
            prompt_content = match.group(2).strip()
            name = metadata.get('name', agent_file.stem)
            prompts[name] = {
                'content': prompt_content,
                'metadata': metadata
            }
        else:
            raise ValueError(f"No frontmatter found in {agent_file}")
    
    return prompts


def parse_subagent_spawn_command(output: str) -> Tuple[str, str]:
    """Parse subagent spawn command to extract subagent name and task.
    
    Args:
        output: Command output from echo "MINI_SWE_AGENT_SPAWN_CHILD::name\ntask"
    
    Returns:
        Tuple of (subagent_name, task) or (None, task) if no specific subagent
    """
    if "::" not in output:
        raise ValueError(f"Invalid spawn command format: {output}")

    lines = output.strip().split('\n')
    if not lines:
        raise ValueError(f"No lines found in {output}")
    
    trigger_line = lines[0]
    subagent_name = trigger_line.split("::", 1)[1]
    task = '\n'.join(lines[1:])
    
    return subagent_name, task
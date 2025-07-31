"""A derivative of the interactive agent.
The system has a root agent that completes the conversation on its own or delegate tasks to subagents.

The semantics of the agents is meant to be simple: when working on a task, if the root agent wants to, it can
sychronously delegate a task to a subagent, one at a time. All agents use the same LM and mode (human, confirm, yolo).
"""

from dataclasses import dataclass, field
from minisweagent.agents.interactive import InteractiveAgent, InteractiveAgentConfig, console, prompt_session
from minisweagent.agents.default import LimitsExceeded
from minisweagent.agents.utils.subagent_loader import load_subagent_prompts, load_subagent_registry, parse_subagent_spawn_command
from typing import Optional
import re

@dataclass
class ManagerAgentConfig(InteractiveAgentConfig):
    subagent_registry: str = field(default_factory=load_subagent_registry)
    """Registry of available subagents loaded from .claude/agents directory."""
    agent_id: str = "ROOT"
    """Unique identifier for this agent instance."""
    parent_agent: Optional['Manager'] = None
    """Reference to parent agent if this is a child agent."""

class Manager(InteractiveAgent):
    SPAWN_TRIGGER = "MINI_SWE_AGENT_SPAWN_CHILD"
    
    def __init__(self, *args, config_class=ManagerAgentConfig, **kwargs):
        super().__init__(*args, config_class=config_class, **kwargs)
        self.agent_id = kwargs.get('agent_id', 'ROOT')
        self.parent_agent = kwargs.get('parent_agent', None)
        self._child_counter = 0

    @property
    def mode(self):
        """Mode is inherited from parent"""
        if self.parent_agent:
            return self.parent_agent.mode
        return self.config.mode
    
    @mode.setter
    def mode(self, value):
        """Setting mode affects the root"""
        if self.parent_agent:
            self.parent_agent.mode = value
        else:
            self.config.mode = value

    def execute_action(self, action: dict) -> dict:
        """Detect spawn trigger and delegate to child"""
        output = super().execute_action(action)
        if self.SPAWN_TRIGGER in action.get("action", ""):
            output_text = output.get("output", "")
            subagent_name, child_task = parse_subagent_spawn_command(output_text)
            return self._spawn_and_run_child(child_task, subagent_name)

        return output

    def _spawn_and_run_child(self, task: str, subagent_name: str) -> dict:
        """Spawn child agent and run it"""
        self._child_counter += 1
        
        child_id = f"{self.agent_id}::{self._child_counter}-{subagent_name}"
        console.print(f"\n[bold cyan]━━━ Spawning {subagent_name} as {child_id} ━━━[/bold cyan]")
        
        console.print(f"[cyan]Task: {task}[/cyan]\n")
        
        child_kwargs = {
            'agent_id': child_id,
            'parent_agent': self,
        }
        
        # Load subagent config directly
        subagent_prompts = load_subagent_prompts()
        if subagent_name in subagent_prompts:
            subagent_data = subagent_prompts[subagent_name]
            child_kwargs['system_template'] = f"{self.config.system_template}\n\n{subagent_data['content']}"
        else:
            raise ValueError(f"Subagent {subagent_name} not found in registry")
        
        child = Manager(
            self.model, 
            self.env, 
            **child_kwargs
        )
        
        exit_status, exit_message = child.run(task)
        
        console.print(f"\n[bold cyan]━━━ {child_id} finished: {exit_status} ━━━[/bold cyan]")
        
        if exit_status == "Submitted":
            return {
                "output": f"Agent {child_id} returned:\n{exit_message}",
                "exit_code": 0
            }
        else:
            return {
                "output": f"Agent {child_id} failed with {exit_status}: {exit_message}",
                "exit_code": 1
            }

    def add_message(self, role: str, content: str):
        """Show agent ID in UI"""
        super(InteractiveAgent, self).add_message(role, content)
        if role == "assistant":
            console.print(
                f"\n[red][bold]{self.agent_id}[/bold] "
                f"(step [bold]{self.model.n_calls}[/bold], "
                f"[bold]${self.model.cost:.2f}[/bold]):[/red]\n",
                end="",
                highlight=False,
            )
        else:
            console.print(f"\n[bold green]{role.capitalize()}[/bold green]:\n", end="", highlight=False)
        console.print(content, highlight=False, markup=False)

    def query(self) -> dict:
        """Use inherited mode"""
        try:
            with console.status("Waiting for the LM to respond..."):
                return super(InteractiveAgent, self).query()
        except LimitsExceeded:
            console.print(
                f"Limits exceeded. Limits: {self.config.step_limit} steps, ${self.config.cost_limit}.\n"
                f"Current spend: {self.model.n_calls} steps, ${self.model.cost:.2f}."
            )
            self.config.step_limit = int(input("New step limit: "))
            self.config.cost_limit = float(input("New cost limit: "))
            return super(InteractiveAgent, self).query()

    def should_ask_confirmation(self, action: str) -> bool:
        """Use inherited mode"""
        return self.mode == "confirm" and not any(re.match(r, action) for r in self.config.whitelist_actions)

    def _prompt_and_handle_special(self, prompt: str) -> str:
        """Override to use inherited mode and clarify global effect"""
        console.print(prompt, end="")
        user_input = prompt_session.prompt("")
        if user_input == "/h":
            console.print(
                f"Current mode: [bold green]{self.mode}[/bold green]\n"
                f"Current agent: [bold green]{self.agent_id}[/bold green]\n"
                f"[bold green]/y[/bold green] to switch to [bold yellow]yolo[/bold yellow] mode\n"
                f"[bold green]/c[/bold green] to switch to [bold yellow]confirmation[/bold yellow] mode\n"
                f"[bold green]/u[/bold green] to switch to [bold yellow]human[/bold yellow] mode\n"
                f"[dim]Mode changes affect all agents in the hierarchy[/dim]\n"
            )
            return self._prompt_and_handle_special(prompt)
        if user_input in self._MODE_COMMANDS_MAPPING:
            if self.mode == self._MODE_COMMANDS_MAPPING[user_input]:
                return self._prompt_and_handle_special(
                    f"[bold red]Already in {self.mode} mode.[/bold red]\n{prompt}"
                )
            self.mode = self._MODE_COMMANDS_MAPPING[user_input]
            console.print(f"Switched to [bold green]{self.mode}[/bold green] mode (applies to all agents).")
            return user_input
        return user_input


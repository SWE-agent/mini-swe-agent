
import ast
import re
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path

from minisweagent.agents.default import AgentConfig, DefaultAgent, LimitsExceeded


@dataclass(unsafe_hash=True)
class MemoryNode:
    id: int
    role: str
    content: str = field(hash=False)
    timestamp: float = field(hash=False)
    summary: str = field(default="", hash=False)
    metadata: dict = field(default_factory=dict, hash=False)
    raw_content: str = field(default="", hash=False, repr=False)
    

class MemorySearchConfig(AgentConfig):
    """Pydantic config — inherits ``system_template``/``instance_template`` from
    ``AgentConfig`` and adds memory-search knobs.

    Note: previously declared as ``@dataclass`` on a Pydantic parent, which
    silently broke default inheritance (``MemorySearchConfig()`` raised
    ``missing 23 required positional arguments``). Switched to Pydantic so
    the agent is actually instantiable.
    """

    system_template: str = ""
    instance_template: str = ""
    token_budget: int = 16000
    max_node_chars: int = 4000
    diversity_lambda: float = 0.7  # MMR: λ for relevance vs (1-λ) for diversity
    w_content: float = 0.5  # relevance weight for lexical overlap
    w_graph: float = 0.5  # relevance weight for graph/file signals
    graph_overlap_weight: float = 0.5  # same-file bonus
    graph_import_weight: float = 0.3  # AST import-relation bonus
    graph_adjacency_weight: float = 0.2  # nearby node-id bonus
    graph_import_regex_multiplier: float = 0.5  # regex fallback confidence penalty
    enable_repo_background_card: bool = True
    repo_background_card_max_chars: int = 2400
    repo_background_refresh_mode: str = "event"
    repo_background_refresh_cooldown_steps: int = 6
    repo_background_refresh_new_files_threshold: int = 20
    repo_background_refresh_new_topdirs_threshold: int = 1
    repo_background_refresh_existing_file_touch_threshold: int = 3
    repo_background_refresh_edit_event_threshold: int = 2
    cold_start_graph_edge_threshold: int = 5
    cold_start_w_content: float = 0.8
    cold_start_w_graph: float = 0.2
    n_recent_fixed: int = 6          # configurable recent window
    n_anchor_nodes: int = 5          # number of recent nodes for multi-anchor
    beam_patience: int = 2           # allow N expansions with negative marginal gain

class MemorySearchAgent(DefaultAgent):
    REPO_CARD_BEGIN = "<!-- REPO_BACKGROUND_CARD:BEGIN -->"
    REPO_CARD_END = "<!-- REPO_BACKGROUND_CARD:END -->"

    def __init__(self, *args, **kwargs):
        # Use MemorySearchConfig by default unless overridden
        kwargs.setdefault('config_class', MemorySearchConfig)
        super().__init__(*args, **kwargs)
        
        self.memory_graph: list[MemoryNode] = []
        self.next_node_id = 0
        self._import_module_cache: dict[int, tuple[set[str], set[str]]] = {}
        self._repo_background_card_text: str = ""
        self._repo_background_card_version: int = 0
        self._repo_background_last_refresh_step: int = -10**9
        self._seen_files_for_card: set[str] = set()
        self._seen_topdirs_for_card: set[str] = set()
        self._file_observation_counts_at_refresh: dict[str, int] = {}
        self._edit_observation_count_at_refresh: int = 0
    
    def _max_node_chars(self) -> int:
        """Per-node character cap, independent of total token budget."""
        return self.config.max_node_chars

    def _compress_content(self, content: str | None, max_chars: int | None = None) -> str:
        """
        Truncate long content while keeping both the head and tail so the model
        can still see useful code/context instead of only the initial docstring.

        Toolcall responses can carry ``content=None`` (only tool_calls), so we
        coerce ``None`` to "" before any string operation.
        """
        if content is None:
            content = ""
        max_chars = max_chars or self._max_node_chars()
        if len(content) <= max_chars:
            return content
        
        head = max_chars // 2
        tail = max_chars - head
        return (
            content[:head]
            + f"\n... [Truncated {len(content) - max_chars} chars] ...\n"
            + content[-tail:]
        )
        
    def add_messages(self, *messages: dict) -> list[dict]:
        # Override the parent (plural) entry point so that *every* message —
        # including system + initial user task seeded by DefaultAgent.run() —
        # is mirrored into our memory graph.
        result = super().add_messages(*messages)
        for msg in messages:
            metadata = self._derive_metadata(msg)
            self._append_graph_node(
                role=msg.get("role", ""),
                content=msg.get("content", ""),
                metadata=metadata,
            )
        return result

    def _derive_metadata(self, msg: dict) -> dict:
        """Build metadata dict for a graph node.

        For user observations we attach ``command``/``is_edit_command``/
        ``filenames`` derived from the previous assistant message. We
        prefer ``extra.actions`` (the structured channel that works for
        both text-based and tool-call models) and fall back to parsing
        the assistant's ```bash``` fence only when ``extra.actions`` is
        unavailable. Without this metadata the file graph and the
        priority-file boost have no input.
        """
        metadata: dict = dict(msg.get("extra") or {})
        # Pre-stash the assistant's own extra (incl. actions) so the
        # *next* user observation can read it back from the graph.
        if msg.get("role") == "assistant" and msg.get("extra"):
            metadata.setdefault("source_extra", msg["extra"])
            return metadata

        # Toolcall mode uses role="tool" for observations; text-based uses "user".
        if msg.get("role") not in ("user", "tool") or not self.memory_graph:
            return metadata

        prev = self.memory_graph[-1]
        if prev.role != "assistant":
            return metadata

        cmd = self._command_from_assistant_node(prev)
        if not cmd:
            return metadata

        metadata.setdefault("command", cmd)
        metadata.setdefault("is_edit_command", self._is_edit_command(cmd))
        filenames = self._extract_filenames(cmd)
        if filenames:
            metadata.setdefault("filenames", filenames)
        return metadata

    @staticmethod
    def _command_from_assistant_node(node: "MemoryNode") -> str:
        """Extract a bash command from an assistant graph node.

        Order of preference:
        1. ``metadata.source_extra.actions`` (structured: tool-call & text-based)
        2. ``raw_content`` ```bash``` fence (text-based fallback)
        """
        md = node.metadata if isinstance(node.metadata, dict) else {}
        extra = md.get("source_extra") or {}
        actions = extra.get("actions") or []
        if actions:
            first = actions[0]
            if isinstance(first, dict):
                # Both schemas: textbased uses {"action": "..."}, toolcall
                # often uses {"arguments": {"cmd": ...}} or similar.
                cmd = (
                    first.get("action")
                    or first.get("cmd")
                    or first.get("command")
                    or (first.get("arguments") or {}).get("cmd")
                    or (first.get("arguments") or {}).get("command")
                )
                if cmd:
                    return cmd.strip()
            elif isinstance(first, str):
                return first.strip()
        # Fallback: rendered assistant text — v2 uses
        # ```mswea_bash_command``` , v1 used ```bash``` . Match both.
        text = node.raw_content or node.content or ""
        match = re.search(
            r"```(?:mswea_bash_command|bash)\s*\n(.*?)\n```", text, re.DOTALL,
        )
        return match.group(1).strip() if match else ""

    def add_message(self, role: str, content: str, **kwargs):
        # Singular helper used by our own code paths. Delegates to add_messages
        # so the graph is populated through a single code path, then attaches
        # extra metadata (which is graph-only, not part of the chat message).
        self.add_messages({"role": role, "content": content})
        metadata = kwargs.get("metadata", {})
        if metadata and self.memory_graph:
            self.memory_graph[-1].metadata = metadata

    def _append_graph_node(self, role: str, content: str | None, metadata: dict):
        # Toolcall responses can have content=None — store as empty string.
        safe_content = content if content is not None else ""
        truncated_content = self._compress_content(safe_content)
        node = MemoryNode(
            id=self.next_node_id,
            role=role,
            content=truncated_content,
            timestamp=time.time(),
            summary=safe_content[:200],
            metadata=metadata,
            raw_content=safe_content,
        )
        self.memory_graph.append(node)
        self.next_node_id += 1

    def _is_edit_command(self, cmd: str) -> bool:
        command = cmd.strip()
        if not command:
            return False

        tokens = command.split()
        tool = tokens[0] if tokens else ""
        edit_tools = {
            "apply_patch",
            "sed",
            "perl",
            "ed",
            "ex",
            "tee",
            "cp",
            "mv",
            "rm",
            "touch",
            "truncate",
            "install",
        }
        if tool in edit_tools:
            return True
        if tool == "git" and len(tokens) > 1 and tokens[1] in {"apply", "checkout", "restore", "mv", "rm"}:
            return True
        if re.search(r"(^|\s)(>|>>)\s*", command):
            return True
        if re.search(r"(^|\s)-i(\s|$)", command):
            return True
        return False

    def _is_valid_file_path(self, s: str) -> bool:
        """Return True only if s looks like an actual file/directory path."""
        if not s or len(s) < 2:
            return False
        # Reject shell operators and redirection tokens
        if s in {'>', '>>', '<', '<<', '|', '&&', '||', ';', '&', '.', '..', '/'}:
            return False
        if s.startswith('<<') or s.startswith('>>'):
            return False
        # Reject ALL-CAPS heredoc terminators (EOF, END, HEREDOC …)
        if re.fullmatch(r'[A-Z_]{2,}', s):
            return False
        # Must contain '.' (extension) or '/' (directory separator) to look path-like
        if '.' not in s and '/' not in s:
            return False
        # Reject tokens that start with shell-special characters
        if s[0] in {'-', '=', '$', '#', '@', '!', '*', '?', '(', ')', '[', ']', '{', '}', '\\'}:
            return False
        return True

    def _extract_filenames(self, cmd: str) -> list[str]:
        """Extract file paths from a command's first line only (avoids heredoc bodies)."""
        # Multi-line commands (heredocs, scripts) - only parse the first line
        first_line = cmd.split('\n')[0].strip() if '\n' in cmd else cmd.strip()
        parts = first_line.split()
        if not parts:
            return []

        tool = parts[0]
        raw_filenames: list[str] = []

        if tool in ["cat", "read_file", "open", "head", "tail", "less", "more"]:
            for part in parts[1:]:
                if part.startswith('-'):
                    continue
                raw_filenames.append(part.strip("'\""))
        elif tool in ["diff", "cmp"]:
            for part in parts[1:]:
                if part.startswith('-'):
                    continue
                raw_filenames.append(part.strip("'\""))
                if len(raw_filenames) >= 2:
                    break
        elif tool == "grep":
            found_pattern = False
            for part in parts[1:]:
                if part.startswith('-'):
                    continue
                if not found_pattern:
                    found_pattern = True
                    continue
                raw_filenames.append(part.strip("'\""))

        return [f for f in raw_filenames if self._is_valid_file_path(f)]

    def _normalize_observed_path(self, path_value: str) -> str:
        raw = str(path_value).strip().strip("'\"")
        if not raw:
            return ""
        path = Path(raw)
        cwd = Path.cwd()
        if path.is_absolute():
            try:
                path = path.relative_to(cwd)
            except ValueError:
                path = Path(path.name)
        normalized = str(path).replace("\\", "/").strip()
        return normalized

    def _node_file_paths(self, node: MemoryNode) -> list[str]:
        if not isinstance(node.metadata, dict):
            return []
        paths = node.metadata.get("filenames", []) or []
        normalized = []
        for path in paths:
            norm = self._normalize_observed_path(path)
            if norm:
                normalized.append(norm)
        return normalized

    def _collect_observed_file_paths(self) -> set[str]:
        paths = set()
        for node in self.memory_graph:
            paths.update(self._node_file_paths(node))
        return paths

    def _extract_topdirs(self, file_paths: set[str]) -> set[str]:
        topdirs = set()
        for path in file_paths:
            parts = Path(path).parts
            if not parts:
                continue
            if len(parts) == 1:
                topdirs.add("(root)")
            else:
                topdirs.add(parts[0])
        return topdirs

    def _strip_repo_background_card(self, system_text: str) -> str:
        pattern = re.compile(
            rf"{re.escape(self.REPO_CARD_BEGIN)}.*?{re.escape(self.REPO_CARD_END)}",
            re.DOTALL,
        )
        stripped = pattern.sub("", system_text).strip()
        return stripped

    def _compose_system_with_card(self, base_system: str, card_text: str) -> str:
        block = f"{self.REPO_CARD_BEGIN}\n{card_text}\n{self.REPO_CARD_END}"
        if not base_system:
            return block
        return f"{base_system}\n\n{block}"

    def _estimate_fixed_chars_with_system(self, candidate_system: str) -> int:
        if not self.memory_graph:
            return len(self._compress_content(candidate_system, self._max_node_chars()))

        total_nodes = len(self.memory_graph)
        n_recent = self.config.n_recent_fixed
        fixed_indices = {0}
        start_recent = max(1, total_nodes - n_recent)
        for i in range(start_recent, total_nodes):
            fixed_indices.add(i)
        fixed_indices.add(total_nodes - 1)

        total_chars = 0
        for idx in fixed_indices:
            if idx < 0 or idx >= total_nodes:
                continue
            if idx == 0:
                raw = candidate_system
            else:
                raw = getattr(self.memory_graph[idx], "raw_content", self.memory_graph[idx].content)
            total_chars += len(self._compress_content(raw, self._max_node_chars()))
        return total_chars

    def _apply_system_content(self, system_text: str):
        if not self.messages or self.messages[0].get("role") != "system":
            return

        self.messages[0]["content"] = system_text
        if not self.memory_graph:
            return

        system_node = self.memory_graph[0]
        if system_node.role != "system":
            return
        system_node.raw_content = system_text
        system_node.content = self._compress_content(system_text, self._max_node_chars())
        system_node.summary = system_text[:200]

    def compute_memory_stats(self) -> dict:
        """Return aggregate signals about the memory graph.

        Designed to feed back into the planner (memory → plan channel).
        Returns a plain dict so callers can map onto their own dataclass
        without an import cycle.
        """
        file_counter = self._observed_file_counter()
        most_file, most_count = (None, 0)
        if file_counter:
            most_file, most_count = file_counter.most_common(1)[0]

        # Repeat-action: count nodes by exact command string.
        cmd_counter: Counter = Counter()
        read_counter: Counter = Counter()
        edit_events = 0
        for node in self.memory_graph:
            md = node.metadata if isinstance(node.metadata, dict) else {}
            cmd = md.get("command")
            if cmd:
                cmd_counter[cmd] += 1
            if md.get("is_edit_command"):
                edit_events += 1
            else:
                for path in self._node_file_paths(node):
                    read_counter[path] += 1

        repeat_action = max(cmd_counter.values(), default=0)
        return {
            "total_nodes": len(self.memory_graph),
            "distinct_files_touched": len(file_counter),
            "most_touched_file": most_file,
            "most_touched_count": most_count,
            "repeat_action_count": repeat_action,
            "edit_event_count": edit_events,
            "file_read_counts": dict(read_counter),
        }

    def _observed_file_counter(self) -> Counter:
        counter = Counter()
        for node in self.memory_graph:
            for path in self._node_file_paths(node):
                counter[path] += 1
        return counter

    def _count_edit_events(self) -> int:
        count = 0
        for node in self.memory_graph:
            if isinstance(node.metadata, dict) and node.metadata.get("is_edit_command"):
                count += 1
        return count

    def _infer_primary_languages(self, file_counter: Counter) -> list[str]:
        ext_map = {
            ".py": "Python",
            ".js": "JavaScript",
            ".ts": "TypeScript",
            ".tsx": "TypeScript",
            ".jsx": "JavaScript",
            ".java": "Java",
            ".go": "Go",
            ".rs": "Rust",
            ".cpp": "C++",
            ".c": "C",
            ".h": "C/C++",
            ".md": "Markdown",
            ".yml": "YAML",
            ".yaml": "YAML",
            ".toml": "TOML",
            ".json": "JSON",
            ".sh": "Shell",
        }

        lang_counter = Counter()
        for path, freq in file_counter.items():
            suffix = Path(path).suffix.lower()
            language = ext_map.get(suffix)
            if language:
                lang_counter[language] += freq

        if not lang_counter:
            return ["Python"]
        return [name for name, _ in lang_counter.most_common(3)]

    def _infer_repo_name(self) -> str:
        """Get the task repo name from env.cwd (the Docker workdir), not host cwd."""
        env_cwd = getattr(self.env, 'cwd', None)
        if env_cwd:
            return Path(env_cwd).name
        return 'unknown'

    def _discover_top_level_layout(self) -> list[str]:
        """Infer top-level dirs from observed file paths (no host filesystem access)."""
        file_paths = self._collect_observed_file_paths()
        top_dirs: set[str] = set()
        for path in file_paths:
            parts = Path(path).parts
            if len(parts) >= 2:
                top_dirs.add(parts[0])
        return sorted(top_dirs)[:10]

    def _discover_key_entrypoints(self) -> list[str]:
        """Infer key directories from most-accessed paths (no host filesystem access)."""
        file_counter = self._observed_file_counter()
        dir_counter: Counter = Counter()
        for path, freq in file_counter.items():
            parts = Path(path).parts
            # Accumulate frequency at each directory depth (up to 3 levels)
            for depth in range(1, min(len(parts), 4)):
                dir_path = '/'.join(parts[:depth])
                dir_counter[dir_path] += freq
        return [p for p, _ in dir_counter.most_common(6)]

    def _build_repo_background_card(self) -> str:
        repo_name = self._infer_repo_name()
        file_counter = self._observed_file_counter()
        languages = self._infer_primary_languages(file_counter)
        top_layout = self._discover_top_level_layout()
        entrypoints = self._discover_key_entrypoints()
        hot_files = [path for path, _ in file_counter.most_common(8)]

        lines = [
            f"Repo background card v{self._repo_background_card_version + 1}",
            "",
            "Project overview:",
            f"- Repository: {repo_name}",
            f"- Primary languages: {', '.join(languages)}",
            "",
            "Top-level layout:",
        ]
        if top_layout:
            lines.extend([f"- {name}" for name in top_layout])
        else:
            lines.append("- (unavailable)")

        lines.extend([
            "",
            "Key entrypoints:",
        ])
        if entrypoints:
            lines.extend([f"- {item}" for item in entrypoints])
        else:
            lines.append("- (none discovered)")

        lines.extend([
            "",
            "Notes:",
            "- File paths are inferred from observed shell commands in this session.",
        ])

        lines.extend([
            "",
            "Likely hot files:",
        ])
        if hot_files:
            lines.extend([f"- {path}" for path in hot_files])
        else:
            lines.append("- (no observed files yet)")

        card_text = "\n".join(lines).strip()
        return self._compress_content(card_text, self.config.repo_background_card_max_chars)

    def _should_refresh_repo_background_card(self) -> bool:
        if not self.config.enable_repo_background_card:
            return False

        if self._repo_background_card_version == 0:
            return True

        if self.config.repo_background_refresh_mode != "event":
            return True

        step = self.n_calls
        if step - self._repo_background_last_refresh_step < self.config.repo_background_refresh_cooldown_steps:
            return False

        current_counter = self._observed_file_counter()
        current_files = set(current_counter.keys())
        current_topdirs = self._extract_topdirs(current_files)
        new_files = current_files - self._seen_files_for_card
        new_topdirs = current_topdirs - self._seen_topdirs_for_card
        existing_file_touch_events = 0
        for path, count in current_counter.items():
            if path not in self._seen_files_for_card:
                continue
            baseline = self._file_observation_counts_at_refresh.get(path, 0)
            if count > baseline:
                existing_file_touch_events += count - baseline

        current_edit_events = self._count_edit_events()
        edit_events_since_refresh = max(0, current_edit_events - self._edit_observation_count_at_refresh)
        return (
            len(new_files) >= self.config.repo_background_refresh_new_files_threshold
            or len(new_topdirs) >= self.config.repo_background_refresh_new_topdirs_threshold
            or existing_file_touch_events >= self.config.repo_background_refresh_existing_file_touch_threshold
            or edit_events_since_refresh >= self.config.repo_background_refresh_edit_event_threshold
        )

    def _ensure_repo_background_card(self):
        if not self.config.enable_repo_background_card:
            return
        if not self.messages or self.messages[0].get("role") != "system":
            return
        if not self._should_refresh_repo_background_card():
            return

        current_files = self._collect_observed_file_paths()
        current_topdirs = self._extract_topdirs(current_files)

        existing_system = self.messages[0]["content"]
        base_system = self._strip_repo_background_card(existing_system)
        full_card = self._build_repo_background_card()
        candidate_system = self._compose_system_with_card(base_system, full_card)

        max_chars_budget = self.config.token_budget * 4
        fixed_safety_limit = int(0.7 * max_chars_budget)

        selected_card = full_card
        selected_system = candidate_system
        estimated_chars = self._estimate_fixed_chars_with_system(selected_system)
        if estimated_chars > fixed_safety_limit:
            half_limit = max(200, self.config.repo_background_card_max_chars // 2)
            fallback_card = self._compress_content(full_card, half_limit)
            fallback_system = self._compose_system_with_card(base_system, fallback_card)
            fallback_chars = self._estimate_fixed_chars_with_system(fallback_system)
            if fallback_chars > fixed_safety_limit:
                return
            selected_card = fallback_card
            selected_system = fallback_system

        self._apply_system_content(selected_system)
        self._repo_background_card_text = selected_card
        self._repo_background_card_version += 1
        self._repo_background_last_refresh_step = self.n_calls
        self._seen_files_for_card.update(current_files)
        self._seen_topdirs_for_card.update(current_topdirs)
        self._file_observation_counts_at_refresh = dict(self._observed_file_counter())
        self._edit_observation_count_at_refresh = self._count_edit_events()
        
    def query(self) -> dict:
        """Override query to context search before calling model.

        Uses the same accounting contract as ``DefaultAgent.query``:
        increments ``self.n_calls`` / ``self.cost`` and raises
        ``LimitsExceeded`` with a structured exit message so the run
        loop can terminate cleanly.
        """
        if 0 < self.config.step_limit <= self.n_calls or 0 < self.config.cost_limit <= self.cost:
            raise LimitsExceeded({
                "role": "exit",
                "content": "LimitsExceeded",
                "extra": {"exit_status": "LimitsExceeded", "submission": ""},
            })

        self._ensure_repo_background_card()
        
        # 1. Construct Context via Search
        selected_nodes = self.construct_context_via_search()
        
        # 2. Re-assemble messages for the LLM
        # We need to preserve system prompt usually, and then the selected nodes
        # sorted by timestamp to maintain readable flow? 
        # Or Just purely selected nodes.
        # For this experiment, let's sort selected nodes by ID (time) to make sense.
        selected_nodes.sort(key=lambda n: n.id)

        # Toolcall mode: don't rewrite messages — the API requires every
        # `tool_calls` block to be followed by matching `tool` messages.
        # See PlanMemAgent.query for the same logic.
        toolcall_mode = any(
            m.get("role") == "tool" or m.get("tool_calls") for m in self.messages
        )
        original_messages = self.messages
        if not toolcall_mode:
            max_chars = self._max_node_chars()
            self.messages = [
                {"role": n.role, "content": self._compress_content(
                    getattr(n, "raw_content", n.content), max_chars,
                )}
                for n in selected_nodes
            ]
        try:
            self.n_calls += 1
            response = self.model.query(self.messages)
        finally:
            if not toolcall_mode:
                self.messages = original_messages

        self.cost += response.get("extra", {}).get("cost", 0.0)
        self.add_messages(response)
        return response

    @dataclass
    class BeamState:
        selected_indices: set[int]
        current_chars: int
        score: float
    
    def construct_context_via_search(self) -> list[MemoryNode]:
        """
        Construct context using Beam Search + MMR to find the optimal set of nodes
        that fits within the token budget and maximizes relevance.
        """
        if not self.memory_graph:
            return []

        # ── Fix 5: configurable recent window ────────────────────────────────
        n_recent = self.config.n_recent_fixed

        # 1. Identify Fixed Nodes (always included regardless of scoring)
        #    • Node 0: system prompt / task description
        #    • Last n_recent nodes: short-term working memory
        fixed_indices: set[int] = {0}
        total_nodes = len(self.memory_graph)
        for i in range(max(1, total_nodes - n_recent), total_nodes):
            fixed_indices.add(i)
        anchor_idx = total_nodes - 1  # still used for file-graph operations
        fixed_indices.add(anchor_idx)

        # Calculate base cost of fixed nodes
        max_chars_budget = self.config.token_budget * 4  # approx 4 chars/token
        base_chars = 0
        fixed_nodes: list[MemoryNode] = []
        for i in sorted(fixed_indices):
            if 0 <= i < total_nodes:
                node = self.memory_graph[i]
                compressed = self._compress_content(
                    getattr(node, "raw_content", node.content), self._max_node_chars()
                )
                base_chars += len(compressed)
                fixed_nodes.append(node)

        if base_chars >= max_chars_budget:
            return sorted(fixed_nodes, key=lambda n: n.id)

        # 2. Pre-compute shared scoring data
        K = 5  # beam width
        node_word_sets = [self._tokenize_content(node.content) for node in self.memory_graph]
        node_file_stems = [self._node_file_stems(node) for node in self.memory_graph]
        ast_import_graph, regex_import_graph = self._build_import_graphs()
        graph_edge_count = (
            sum(len(v) for v in ast_import_graph.values())
            + sum(len(v) for v in regex_import_graph.values())
        )

        # ── Fix 4: IDF weights ───────────────────────────────────────────────
        idf = self._build_idf_weights()

        # ── Fix 3: multi-anchor word set ─────────────────────────────────────
        # Union of the last n_anchor recent nodes + node 1 (task description).
        # This prevents the anchor from being stuck on the last unrelated command.
        n_anchor = min(self.config.n_anchor_nodes, total_nodes - 1)
        anchor_words: set[str] = set()
        for i in range(max(1, total_nodes - n_anchor), total_nodes):
            anchor_words |= node_word_sets[i]
        if total_nodes > 1:
            anchor_words |= node_word_sets[1]  # task description always included

        # File-graph anchor: union of recent nodes' file stems
        anchor_file_stems: set[str] = set()
        for i in range(max(1, total_nodes - n_anchor), total_nodes):
            anchor_file_stems |= node_file_stems[i]
        # Fallback to _collect_anchor_file_stems if the above is empty
        if not anchor_file_stems:
            anchor_file_stems = self._collect_anchor_file_stems(
                anchor_idx, fixed_indices, node_file_stems
            )

        related_reference_indices = self._build_related_reference_indices(
            anchor_file_stems, node_file_stems, ast_import_graph, regex_import_graph
        )
        if graph_edge_count < self.config.cold_start_graph_edge_threshold or not anchor_file_stems:
            effective_w_content = self.config.cold_start_w_content
            effective_w_graph = self.config.cold_start_w_graph
        else:
            effective_w_content = self.config.w_content
            effective_w_graph = self.config.w_graph

        # 3. Beam Search
        beams = [self.BeamState(fixed_indices.copy(), base_chars, 0.0)]
        best_beam = beams[0]
        candidate_pool = [i for i in range(total_nodes) if i not in fixed_indices]
        max_depth = 10

        # ── Fix 6: patience for negative-gain expansions ──────────────────────
        patience_counter = 0

        for _ in range(max_depth):
            candidates_next_round = []
            expanded_any = False
            best_next_gain = float("-inf")

            for beam in beams:
                selected_non_fixed = beam.selected_indices - fixed_indices
                for cand_idx in candidate_pool:
                    if cand_idx in beam.selected_indices:
                        continue

                    cand_node = self.memory_graph[cand_idx]
                    cand_chars = len(self._compress_content(
                        getattr(cand_node, "raw_content", cand_node.content),
                        self._max_node_chars(),
                    ))

                    if beam.current_chars + cand_chars <= max_chars_budget:
                        cand_words = node_word_sets[cand_idx]
                        # IDF-weighted relevance (Fix 4)
                        lexical_relevance = self.calculate_content_score_optimized(
                            cand_words, anchor_words, idf
                        )
                        graph_relevance = self.calculate_graph_score(
                            cand_idx=cand_idx,
                            anchor_file_stems=anchor_file_stems,
                            node_file_stems=node_file_stems,
                            ast_import_graph=ast_import_graph,
                            regex_import_graph=regex_import_graph,
                            related_reference_indices=related_reference_indices,
                        )
                        relevance = (
                            effective_w_content * lexical_relevance
                            + effective_w_graph * graph_relevance
                        )

                        # IDF-weighted redundancy (Fix 4)
                        redundancy = 0.0
                        for selected_idx in selected_non_fixed:
                            sim = self._node_similarity(
                                cand_words, node_word_sets[selected_idx], idf
                            )
                            redundancy = max(redundancy, sim)

                        lam = self.config.diversity_lambda
                        marginal_gain = lam * relevance - (1 - lam) * redundancy
                        best_next_gain = max(best_next_gain, marginal_gain)

                        candidates_next_round.append(self.BeamState(
                            beam.selected_indices | {cand_idx},
                            beam.current_chars + cand_chars,
                            beam.score + marginal_gain,
                        ))
                        expanded_any = True

            if not expanded_any:
                break

            # ── Fix 6: patience instead of immediate stop on negative gain ────
            if best_next_gain <= 0:
                patience_counter += 1
                if patience_counter >= self.config.beam_patience:
                    break
            else:
                patience_counter = 0

            # Prune to top-K unique beams
            candidates_next_round.sort(key=lambda s: s.score, reverse=True)
            new_beams: list[MemorySearchAgent.BeamState] = []
            seen_sets: set[frozenset] = set()
            for cand in candidates_next_round:
                fs = frozenset(cand.selected_indices)
                if fs not in seen_sets:
                    seen_sets.add(fs)
                    new_beams.append(cand)
                    if len(new_beams) >= K:
                        break
            if not new_beams:
                break
            beams = new_beams
            if beams[0].score > best_beam.score:
                best_beam = beams[0]

        # 4. Final Selection
        print(f"Beam Search Selected {len(best_beam.selected_indices)} nodes with score {best_beam.score:.2f}")
        selected_nodes = [self.memory_graph[i] for i in best_beam.selected_indices]
        
        return selected_nodes

    def _tokenize_content(self, content: str) -> set[str]:
        """Tokenize content for relevance and redundancy scoring."""
        return set(re.sub(r'\W+', ' ', content.lower()).split())

    def _node_filenames(self, node: MemoryNode) -> list[str]:
        normalized = []
        for name in self._node_file_paths(node):
            normalized.append(Path(name).name)
        return normalized

    def _node_file_stems(self, node: MemoryNode) -> set[str]:
        stems = set()
        for filename in self._node_filenames(node):
            stem = Path(filename).stem.lower()
            if stem:
                stems.add(stem)
        return stems

    def _extract_output_body(self, text: str) -> str:
        """Best-effort extraction for templated observations."""
        if text.startswith("Observation:"):
            text = text[len("Observation:"):].strip()
        match = re.search(r"<output>\s*(.*?)\s*</output>", text, re.DOTALL)
        if match:
            return match.group(1)
        return text

    def _strip_line_numbers(self, text: str) -> str:
        lines = text.splitlines()
        cleaned = [re.sub(r"^\s*\d+\s+", "", line) for line in lines]
        return "\n".join(cleaned)

    def _module_tokens(self, module_name: str) -> set[str]:
        tokens = set()
        for token in module_name.split("."):
            token = token.strip().lower()
            if token:
                tokens.add(token)
        return tokens

    def _extract_import_modules_from_source(self, source: str) -> tuple[set[str], bool]:
        try:
            tree = ast.parse(source)
        except (SyntaxError, ValueError):
            return set(), False

        modules = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    modules.update(self._module_tokens(alias.name))
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    modules.update(self._module_tokens(node.module))
                for alias in node.names:
                    if alias.name != "*":
                        modules.update(self._module_tokens(alias.name))
        return modules, True

    def _extract_import_modules_regex(self, source: str) -> set[str]:
        """
        Conservative regex fallback used only when AST parsing fails.
        This improves recall on noisy/incomplete observation snippets.
        """
        modules = set()
        for raw_line in source.splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            # Drop trailing comment to reduce accidental matches.
            line = line.split("#", 1)[0].strip()
            if not line:
                continue
            # Strip common REPL/shell noise prefixes.
            if ">>>" in line:
                line = line.split(">>>")[-1].strip()
            line = re.sub(r"^\$+\s*", "", line)
            line = re.sub(r"^\d+\s+", "", line)
            if not line:
                continue

            from_match = re.match(r"^from\s+([A-Za-z0-9_\.]+)\s+import\s+(.+)$", line)
            if from_match:
                modules.update(self._module_tokens(from_match.group(1)))
                imports_part = from_match.group(2)
                for part in imports_part.split(","):
                    name = part.strip()
                    if not name or name == "*":
                        continue
                    # Handle alias form "foo as bar".
                    name = name.split(" as ", 1)[0].strip()
                    modules.update(self._module_tokens(name))
                continue

            import_match = re.match(
                r"^import\s+([A-Za-z_][A-Za-z0-9_\.]*(?:\s+as\s+[A-Za-z_][A-Za-z0-9_]*)?"
                r"(?:\s*,\s*[A-Za-z_][A-Za-z0-9_\.]*(?:\s+as\s+[A-Za-z_][A-Za-z0-9_]*)?)*)$",
                line,
            )
            if import_match:
                for part in import_match.group(1).split(","):
                    name = part.strip()
                    if not name:
                        continue
                    name = name.split(" as ", 1)[0].strip()
                    modules.update(self._module_tokens(name))
        return modules

    def _node_import_modules(self, node: MemoryNode) -> tuple[set[str], set[str]]:
        """
        Extract import module tokens from python file observations.
        Returns:
        - ast_modules: high-confidence modules from AST parsing
        - regex_modules: fallback modules from regex, only from AST-failed snippets
        """
        cache_key = node.id
        if cache_key in self._import_module_cache:
            return self._import_module_cache[cache_key]

        filenames = self._node_filenames(node)
        if not any(name.lower().endswith(".py") for name in filenames):
            empty = (set(), set())
            self._import_module_cache[cache_key] = empty
            return empty

        raw = getattr(node, "raw_content", node.content)
        candidates = [
            raw,
            self._extract_output_body(raw),
            self._strip_line_numbers(self._extract_output_body(raw)),
        ]

        ast_modules = set()
        regex_modules = set()
        for candidate in candidates:
            if not candidate.strip():
                continue
            parsed_modules, parsed_ok = self._extract_import_modules_from_source(candidate)
            if parsed_ok and parsed_modules:
                ast_modules.update(parsed_modules)
                break
            if not parsed_ok:
                regex_modules.update(self._extract_import_modules_regex(candidate))

        result = (ast_modules, regex_modules if not ast_modules else set())
        self._import_module_cache[cache_key] = result
        return result

    def _build_import_graphs(self) -> tuple[dict[str, set[str]], dict[str, set[str]]]:
        """
        Build separate import graphs:
        - ast_import_graph: high-confidence edges from AST
        - regex_import_graph: fallback edges from regex when AST fails
        """
        ast_import_graph: dict[str, set[str]] = defaultdict(set)
        regex_import_graph: dict[str, set[str]] = defaultdict(set)
        for node in self.memory_graph:
            stems = self._node_file_stems(node)
            if not stems:
                continue
            ast_modules, regex_modules = self._node_import_modules(node)
            if not ast_modules and not regex_modules:
                continue
            for stem in stems:
                if ast_modules:
                    ast_import_graph[stem].update(ast_modules)
                if regex_modules:
                    regex_import_graph[stem].update(regex_modules)
        return dict(ast_import_graph), dict(regex_import_graph)

    def _has_import_relation(
        self, files_a: set[str], files_b: set[str], import_graph: dict[str, set[str]]
    ) -> bool:
        for file_a in files_a:
            if import_graph.get(file_a, set()) & files_b:
                return True
        for file_b in files_b:
            if import_graph.get(file_b, set()) & files_a:
                return True
        return False

    def _import_relation_score(
        self,
        files_a: set[str],
        files_b: set[str],
        ast_import_graph: dict[str, set[str]],
        regex_import_graph: dict[str, set[str]],
    ) -> float:
        if self._has_import_relation(files_a, files_b, ast_import_graph):
            return 1.0
        if self._has_import_relation(files_a, files_b, regex_import_graph):
            return self.config.graph_import_regex_multiplier
        return 0.0

    def _collect_anchor_file_stems(
        self, anchor_idx: int, fixed_indices: set[int], node_file_stems: list[set[str]]
    ) -> set[str]:
        anchor_files = set(node_file_stems[anchor_idx])
        if anchor_files:
            return anchor_files

        # Fallback: anchor may be plain user text with no filename metadata.
        for idx in fixed_indices:
            anchor_files.update(node_file_stems[idx])
        return anchor_files

    def _build_related_reference_indices(
        self,
        anchor_file_stems: set[str],
        node_file_stems: list[set[str]],
        ast_import_graph: dict[str, set[str]],
        regex_import_graph: dict[str, set[str]],
    ) -> set[int]:
        if not anchor_file_stems:
            return set()

        related = set()
        for idx, stems in enumerate(node_file_stems):
            if not stems:
                continue
            has_relation = (
                self._import_relation_score(stems, anchor_file_stems, ast_import_graph, regex_import_graph) > 0
            )
            if stems & anchor_file_stems or has_relation:
                related.add(idx)
        return related

    def _id_adjacency_score(self, cand_idx: int, reference_indices: set[int]) -> float:
        if not reference_indices:
            return 0.0
        min_distance = min(abs(cand_idx - ref_idx) for ref_idx in reference_indices)
        return 1.0 / (1.0 + min_distance)

    def calculate_graph_score(
        self,
        *,
        cand_idx: int,
        anchor_file_stems: set[str],
        node_file_stems: list[set[str]],
        ast_import_graph: dict[str, set[str]],
        regex_import_graph: dict[str, set[str]],
        related_reference_indices: set[int],
    ) -> float:
        """
        File-graph relevance in [0, 1]:
        - same-file overlap
        - AST import relation
        - node-id adjacency to file-related references
        """
        cand_files = node_file_stems[cand_idx]

        overlap_score = 0.0
        import_score = 0.0
        if cand_files and anchor_file_stems:
            if cand_files & anchor_file_stems:
                overlap_score = 1.0
            import_score = self._import_relation_score(
                cand_files, anchor_file_stems, ast_import_graph, regex_import_graph
            )

        adjacency_score = self._id_adjacency_score(cand_idx, related_reference_indices)

        w_overlap = self.config.graph_overlap_weight
        w_import = self.config.graph_import_weight
        w_adjacency = self.config.graph_adjacency_weight
        total = w_overlap + w_import + w_adjacency
        if total <= 0:
            return 0.0

        return (
            w_overlap * overlap_score
            + w_import * import_score
            + w_adjacency * adjacency_score
        ) / total

    def _build_idf_weights(self) -> dict[str, float]:
        """Compute IDF weights from all memory nodes (smoothed log-IDF).

        High-frequency tokens (self, import, return) get low weight;
        rare identifiers (TimeSeries, _required_columns) get high weight.
        """
        import math
        N = len(self.memory_graph)
        if N == 0:
            return {}
        df: Counter = Counter()
        for node in self.memory_graph:
            for w in self._tokenize_content(node.content):
                df[w] += 1
        # Smoothed IDF: log((N+1)/(df+1)) — avoids division by zero and stays positive
        return {word: math.log((N + 1) / (freq + 1)) for word, freq in df.items()}

    def calculate_content_score_optimized(
        self,
        node_words: set[str],
        anchor_words: set[str],
        idf: dict[str, float] | None = None,
    ) -> float:
        """IDF-weighted containment: fraction of anchor word *weight* covered by node."""
        if not anchor_words:
            return 0.0
        intersection = node_words & anchor_words
        if idf is None:
            return len(intersection) / len(anchor_words)
        matched_weight = sum(idf.get(w, 1.0) for w in intersection)
        total_anchor_weight = sum(idf.get(w, 1.0) for w in anchor_words)
        return matched_weight / total_anchor_weight if total_anchor_weight > 0 else 0.0

    def _node_similarity(
        self,
        words_a: set[str],
        words_b: set[str],
        idf: dict[str, float] | None = None,
    ) -> float:
        """IDF-weighted Jaccard similarity between two token sets."""
        if not words_a or not words_b:
            return 0.0
        intersection = words_a & words_b
        union = words_a | words_b
        if idf is None:
            return len(intersection) / len(union)
        matched = sum(idf.get(w, 1.0) for w in intersection)
        total = sum(idf.get(w, 1.0) for w in union)
        return matched / total if total > 0 else 0.0

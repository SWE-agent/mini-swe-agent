"""LLM query detection rules R1-R3 (see the doc "LLM 호출 탐지 규칙 (R1–R3)").

R1  official SDK method chains     (suffix match on the call's attribute chain, no import needed)
R2  intermediary libraries         (method name, only when the file imports that library)
R3  raw HTTP                       (every post/request/send/stream/urlopen when the file imports an HTTP library)

classify(call, imports, cls_strings) -> (rule, pattern, detail) | None
Policy: false positives are allowed, false negatives are not.
"""

import ast
import re

# ---------------------------------------------------------------- rules
# R1. Official SDK method chains (suffix match on the call's attribute chain).
#     Source: sdk_inventory (anthropic 1.13.0, openai 3.31.0, google-genai 2.30.1) + legacy names.
SDK_CHAINS = {
    "anthropic": [
        "messages.create", "messages.stream", "messages.parse", "messages.count_tokens",
        "messages.batches.create", "messages.tool_runner",
        "sessions.create", "sessions.events.send", "sessions.events.stream", "sessions.events.list",
        "sessions.events.tool_runner", "sessions.threads.events.stream", "sessions.threads.events.list",
        "deployments.run", "environments.work.poller", "environments.work.worker",
        "completions.create",
    ],
    "google-genai": [
        "models.generate_content", "models.generate_content_stream", "chats.create",
        "send_message", "send_message_stream", "live.connect", "send_client_content",
        "send_realtime_input", "send_tool_response", "interactions.create", "batches.create",
        "generate_content", "start_chat",  # legacy google-generativeai / vertexai
    ],
    "openai": [
        "responses.create", "responses.stream", "responses.parse", "responses.compact", "responses.connect",
        "chat.completions.create", "chat.completions.parse", "chat.completions.stream", "completions.create",
        "threads.create_and_run", "threads.create_and_run_poll", "threads.create_and_run_stream",
        "runs.create", "runs.create_and_poll", "runs.create_and_stream", "runs.stream", "runs.poll",
        "runs.submit_tool_outputs", "runs.submit_tool_outputs_and_poll", "runs.submit_tool_outputs_stream",
        "realtime.connect", "realtime.calls.create", "realtime.calls.accept", "client_secrets.create",
        "realtime.sessions.create", "live.connect", "agents.sessions.create", "agents.sessions.stream",
        "sessions.events.create", "sessions.turns", "chatkit.sessions.create", "evals.runs.create",
        "ChatCompletion.create", "Completion.create",  # v0
    ],
}
IGNORED_SEGMENTS = {"with_raw_response", "with_streaming_response", "aio", "beta"}

# R2. Intermediary libraries: method names that count only when the file imports that library.
LIB_RULES = {
    "litellm": ["completion", "acompletion", "responses", "aresponses", "text_completion", "atext_completion"],
    "langchain": ["invoke", "ainvoke", "stream", "astream", "batch", "abatch", "generate", "agenerate", "predict"],
    "langchain_core": ["invoke", "ainvoke", "stream", "astream", "batch", "abatch", "generate", "agenerate"],
    "llama_index": ["complete", "acomplete", "chat", "achat", "stream_complete", "stream_chat"],
    "smolagents": ["generate", "generate_stream"],
    "ollama": ["chat", "generate"],
    "mistralai": ["complete", "complete_async", "stream", "stream_async"],
    "cohere": ["chat", "chat_stream"],
    "boto3": ["invoke_model", "invoke_model_with_response_stream", "converse", "converse_stream"],
    "huggingface_hub": ["chat_completion", "text_generation"],
    "oci": ["chat", "generate_text"],  # Oracle Cloud Generative AI
    "groq": ["create"],                # groq SDK is OpenAI-shaped; R1 usually catches it already
    # local inference (no network, still an LLM call)
    "transformers": ["generate", "__call__"],
    "vllm": ["generate", "chat"],
    "mlx_lm": ["generate", "stream_generate"],
    "llama_cpp": ["create_chat_completion", "create_completion", "__call__"],
}

# Third-party libraries a reviewer has checked and confirmed do NOT call an LLM (e.g. "yaml", "pydantic", "rich").
# Every other third-party import that no rule covers is listed in unreviewed_imports.tsv, so a new SDK
# (like `oci` in browser-use) is seen by a person instead of being silently missed. Add names here after review.
REVIEWED_NON_LLM: set[str] = set()
RULE_LIBS = {"openai", "anthropic", "google", "vertexai"}  # R1 SDK packages; R2/R3 packages are added below

# R3. Raw HTTP: every post/request through these libraries is flagged; the URL is resolved when possible.
HTTP_LIBS = ("requests", "httpx", "aiohttp", "urllib")
HTTP_METHODS = {"post", "request", "send", "urlopen", "stream"}
LLM_HOSTS = re.compile(
    r"api\.openai\.com|api\.anthropic\.com|generativelanguage\.googleapis\.com|aiplatform\.googleapis\.com|"
    r"openrouter\.ai|requesty\.ai|portkey\.ai|api\.mistral\.ai|api\.cohere|api\.groq\.com|api\.together|"
    r"api\.deepseek\.com|api\.x\.ai|bedrock-runtime|/chat/completions|/v1/messages|/responses|:generateContent"
)


def chain_of(func: ast.AST) -> list[str]:
    """`self.client.beta.chat.completions.create` -> ['self','client','chat','completions','create']."""
    parts = []
    while isinstance(func, ast.Attribute):
        parts.append(func.attr)
        func = func.value
    if isinstance(func, ast.Name):
        parts.append(func.id)
    elif isinstance(func, ast.Call):
        parts.append(ast.unparse(func.func) + "()")
    return [p for p in reversed(parts) if p not in IGNORED_SEGMENTS]


def imported_roots(tree: ast.Module) -> dict[str, str]:
    """local name -> top-level package it comes from (also for `from x import y`)."""
    out = {}
    for n in ast.walk(tree):
        if isinstance(n, ast.Import):
            for a in n.names:
                out[(a.asname or a.name).split(".")[0]] = a.name.split(".")[0]
        elif isinstance(n, ast.ImportFrom) and n.module:
            for a in n.names:
                out[a.asname or a.name] = n.module.split(".")[0]
    return out


def self_string_attrs(cls: ast.ClassDef | None) -> dict[str, str]:
    """`self.x = "https://..."` (or f-string) anywhere in the class -> {'x': literal}."""
    out = {}
    if cls is None:
        return out
    for n in ast.walk(cls):
        if isinstance(n, (ast.Assign, ast.AnnAssign)) and n.value is not None:
            targets = n.targets if isinstance(n, ast.Assign) else [n.target]
            for t in targets:
                if isinstance(t, ast.Attribute) and isinstance(t.value, ast.Name) and t.value.id == "self":
                    s = " ".join(x.value for x in ast.walk(n.value) if isinstance(x, ast.Constant) and isinstance(x.value, str))
                    if s:
                        out[t.attr] = s
    return out


RULES = {"R1", "R2", "R3"}


def references(node: ast.AST):
    """Attributes used as a VALUE, not called on the spot: `retryer(self.client.chat.completions.create, **kw)`,
    `functools.partial(litellm.completion, ...)`, `fn = client.messages.create`. Each is wrapped in a fake call
    so classify() can judge it; the rule is reported with a `-ref` suffix."""
    called = {id(c.func) for c in ast.walk(node) if isinstance(c, ast.Call)}
    inner = {id(a.value) for a in ast.walk(node) if isinstance(a, ast.Attribute)}  # skip prefixes of longer chains
    for a in ast.walk(node):
        if isinstance(a, ast.Attribute) and id(a) not in called and id(a) not in inner:
            fake = ast.Call(func=a, args=[], keywords=[])
            ast.copy_location(fake, a)
            yield fake


def classify(call: ast.Call, imports: dict[str, str], cls_strings: dict[str, str]):
    ch = chain_of(call.func)
    if not ch:
        return None
    dotted = ".".join(ch)
    hits = [(len(c.split(".")), sdk, c) for sdk, chains in SDK_CHAINS.items() for c in chains
            if ch[-len(c.split(".")):] == c.split(".")]
    if hits and "R1" in RULES:
        best = max(h[0] for h in hits)
        top = sorted({(sdk, c) for n, sdk, c in hits if n == best})
        return ("R1-sdk", "/".join(sdk for sdk, _ in top) + ":" + top[0][1], dotted)
    root_lib = imports.get(ch[0])
    for lib, names in (LIB_RULES.items() if "R2" in RULES else []):
        if ch[-1] in names and (root_lib == lib or lib in imports.values()):
            if root_lib == lib or len(ch) > 1:
                return ("R2-lib", f"{lib}:{ch[-1]}", dotted)
    if "R3" in RULES and ch[-1] in HTTP_METHODS and any(l in imports.values() for l in HTTP_LIBS):
        url = ""
        args = list(call.args) + [k.value for k in call.keywords if k.arg in ("url",)]
        for a in args:
            if isinstance(a, ast.Constant) and isinstance(a.value, str):
                url = a.value
            elif isinstance(a, ast.Attribute) and isinstance(a.value, ast.Name) and a.value.id == "self":
                url = cls_strings.get(a.attr, f"self.{a.attr}")
            if url:
                break
        tag = "LLM host" if LLM_HOSTS.search(url) else "host unknown"
        return ("R3-http", f"{ch[-1]} ({tag})", f"{dotted}  url={url[:80]}")
    return None
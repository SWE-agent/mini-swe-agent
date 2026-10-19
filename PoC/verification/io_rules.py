"""I/O detection rules for taint-style pruning: what an agent takes from the LLM, executes, and reads from the user.

LLM   R1-R3   delegated to llm_rules.classify (colleague's rules, unchanged)
EXTRACT       tool calls taken out of an LLM response
  E1  native tool-call fields     `.tool_calls`, `.function_call(s)`, "tool_use" / "function_call" blocks
  E2  text parsing                re.* / json.loads / yaml.*load / xml parsing (only counted near an LLM call,
                                  see io_audit.py: alone they are far too common)
  E3  structured output           response_format= / output_format= / output_type= / response_model= / text_format=,
                                  with_structured_output(), model_validate_json()
TOOL          where an action is actually executed
  T1  process execution           subprocess.*, os.system/popen/exec*/spawn*, asyncio.create_subprocess_*,
                                  pexpect, pty.spawn, docker `exec_run`, paramiko `exec_command`
      T1-remote                   sandbox / remote SDKs, only in files importing them: swerex, contree_sdk, e2b,
                                  modal, daytona, docker SDK, kubernetes exec, fabric / invoke
  T2  code execution              exec / eval / compile (builtins), runpy.*, code.Interactive*
  T3  tool-call conventions       call_tool / execute_tool(_call) / execute_action / run_tool / invoke_tool,
                                  or invoke/run/__call__/forward on something named *tool*
  T4  dynamic dispatch            registry[name](...) / getattr(obj, name)(...) with a non-literal name
INPUT
  I1  user input                  input(), sys.stdin.read*, getpass, prompt_toolkit prompt, rich Prompt/Confirm.ask,
                                  click/typer prompt/confirm, questionary *.ask

T5 (side effects such as browser automation, file writes, non-LLM HTTP) is intentionally left out for now.
Policy (same as llm_rules): false positives are allowed, false negatives are not. Every list is meant to be
reviewed by a person; add or remove names here, not in the driver.
"""

import ast

import llm_rules

# ---------------------------------------------------------------- EXTRACT
E1_ATTRS = {"tool_calls", "function_call", "function_calls", "tool_use"}
E1_STRINGS = {"tool_use", "function_call", "tool_calls", "tool_call"}
E2_CALLS = {  # (module root, last name)
    ("re", "findall"), ("re", "finditer"), ("re", "search"), ("re", "match"), ("re", "fullmatch"),
    ("json", "loads"), ("yaml", "safe_load"), ("yaml", "load"), ("yaml", "full_load"),
    ("xml", "fromstring"), ("xml", "parse"), ("xmltodict", "parse"),
}
E3_KEYWORDS = {"response_format", "output_format", "output_type", "response_model", "text_format"}
E3_METHODS = {"with_structured_output", "model_validate_json", "parse_raw"}

# ---------------------------------------------------------------- TOOL
T1_CALLS = {
    "subprocess": {"run", "Popen", "call", "check_call", "check_output", "getoutput", "getstatusoutput"},
    "os": {"system", "popen", "execv", "execve", "execvp", "execvpe", "execl", "execle", "execlp", "execlpe",
           "spawnv", "spawnve", "spawnvp", "spawnvpe", "spawnl", "spawnle", "spawnlp", "spawnlpe", "posix_spawn",
           "posix_spawnp"},
    "asyncio": {"create_subprocess_exec", "create_subprocess_shell"},
    "pexpect": {"spawn", "run", "popen_spawn"},
    "pty": {"spawn"},
}
T1_METHODS = {"exec_run", "exec_command"}  # docker SDK container.exec_run, paramiko client.exec_command
# T1-remote: sandbox / remote-execution SDKs. Names are generic (run, execute), so like R2 they count only when the
# file imports that library (`self.deployment.runtime.execute(...)` in a file that imports swerex).
T1_REMOTE = {
    "swerex": {"execute", "run_in_session"},  # SWE-ReX runtime (docker / modal / remote deployments)
    "contree_sdk": {"run"},  # ContreeSync session.run(...)
    "e2b": {"run", "run_code"},  # sandbox.commands.run / run_code
    "e2b_code_interpreter": {"run", "run_code"},
    "modal": {"exec"},  # modal.Sandbox.exec
    "daytona": {"exec", "code_run", "execute_command"},  # sandbox.process.exec / code_run
    "daytona_sdk": {"exec", "code_run", "execute_command"},
    "docker": {"exec_run", "run"},  # docker SDK containers.run / container.exec_run
    "kubernetes": {"connect_get_namespaced_pod_exec"},
    "fabric": {"run", "sudo"},  # fabric Connection.run
    "invoke": {"run", "sudo"},
}
T2_BUILTINS = {"exec", "eval", "compile"}
T2_CALLS = {"runpy": {"run_path", "run_module"}, "code": {"InteractiveInterpreter", "InteractiveConsole", "interact"}}
T3_METHODS = {"call_tool", "execute_tool", "execute_tool_call", "execute_action", "run_tool", "invoke_tool"}
T3_TOOLISH = {"invoke", "ainvoke", "run", "arun", "__call__", "forward", "call", "acall"}

# ---------------------------------------------------------------- INPUT
I1_BUILTINS = {"input"}
I1_CALLS = {
    "sys": {"read", "readline", "readlines"},  # sys.stdin.*
    "getpass": {"getpass"},
    "prompt_toolkit": {"prompt", "prompt_async"},
    "rich": {"ask"},  # Prompt.ask / Confirm.ask / IntPrompt.ask
    "click": {"prompt", "confirm"},
    "typer": {"prompt", "confirm"},
    "questionary": {"ask", "unsafe_ask", "ask_async"},
}

CATEGORIES = ("LLM", "EXTRACT", "TOOL", "INPUT")


def _root(ch: list[str], imports: dict[str, str]) -> str | None:
    """Top-level package the call starts from (`sp.run` with `import subprocess as sp` -> 'subprocess')."""
    return imports.get(ch[0]) if ch else None


def classify(call: ast.Call, imports: dict[str, str], strings: dict[str, str]):
    """-> (category, rule, pattern, detail) | None. E2 hits are returned too; the driver decides where they count."""
    if hit := llm_rules.classify(call, imports, strings):
        return ("LLM", *hit)
    ch = llm_rules.chain_of(call.func)
    if not ch:
        return _dynamic(call)
    dotted, last, root = ".".join(ch), ch[-1], _root(ch, imports)
    builtin = len(ch) == 1 and ch[0] not in imports

    if builtin and last in I1_BUILTINS:
        return ("INPUT", "I1-input", last, dotted)
    if root == "sys" and "stdin" in ch and last in I1_CALLS["sys"]:
        return ("INPUT", "I1-input", f"sys.stdin:{last}", dotted)
    for lib, names in I1_CALLS.items():
        # like R2: the method counts when the file imports the library (`session = PromptSession(); session.prompt()`)
        if lib != "sys" and last in names and (root == lib or (lib in imports.values() and len(ch) > 1)):
            return ("INPUT", "I1-input", f"{lib}:{last}", dotted)
    if last in {"prompt", "prompt_async"} and any("session" in p.lower() for p in ch[:-1]):
        return ("INPUT", "I1-input", "*session*.prompt", dotted)  # prompt_toolkit PromptSession passed around

    if root in T1_CALLS and last in T1_CALLS[root]:
        return ("TOOL", "T1-process", f"{root}:{last}", dotted)
    if last in T1_METHODS:
        return ("TOOL", "T1-process", last, dotted)
    for lib, names in T1_REMOTE.items():
        if last in names and (root == lib or (lib in imports.values() and len(ch) > 1)):
            return ("TOOL", "T1-remote", f"{lib}:{last}", dotted)
    if builtin and last in T2_BUILTINS:
        return ("TOOL", "T2-code", last, dotted)
    if root in T2_CALLS and last in T2_CALLS[root]:
        return ("TOOL", "T2-code", f"{root}:{last}", dotted)
    if last in T3_METHODS:
        return ("TOOL", "T3-toolcall", last, dotted)
    if last in T3_TOOLISH and any("tool" in p.lower() for p in ch[:-1]):
        return ("TOOL", "T3-toolcall", f"*tool*.{last}", dotted)

    if (root, last) in E2_CALLS or ("xml" in (root or "") and last in {"fromstring", "parse"}):
        return ("EXTRACT", "E2-parse", f"{root}:{last}", dotted)
    if last in E3_METHODS or any(k.arg in E3_KEYWORDS for k in call.keywords):
        kw = next((k.arg for k in call.keywords if k.arg in E3_KEYWORDS), last)
        return ("EXTRACT", "E3-structured", kw, dotted)
    return _dynamic(call)


def _dynamic(call: ast.Call):
    f = call.func
    if isinstance(f, ast.Subscript) and not isinstance(f.slice, ast.Constant):
        return ("TOOL", "T4-dispatch", "x[name](...)", ast.unparse(f))
    if (
        isinstance(f, ast.Call)
        and isinstance(f.func, ast.Name)
        and f.func.id == "getattr"
        and len(f.args) >= 2
        and not isinstance(f.args[1], ast.Constant)
    ):
        return ("TOOL", "T4-dispatch", "getattr(x, name)(...)", ast.unparse(f))
    return None


def extract_fields(node: ast.AST):
    """E1: tool-call fields read from a response. Yields (ast node, pattern)."""
    for n in ast.walk(node):
        if isinstance(n, ast.Attribute) and n.attr in E1_ATTRS:
            yield n, f".{n.attr}"
        elif isinstance(n, ast.Constant) and isinstance(n.value, str) and n.value in E1_STRINGS:
            yield n, repr(n.value)

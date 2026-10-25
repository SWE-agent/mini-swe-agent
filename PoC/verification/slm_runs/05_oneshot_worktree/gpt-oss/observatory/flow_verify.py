# /// script
# requires-python = ">=3.10"
# dependencies = []
# ///
"""Static verification of an agent's logging from two kinds of anchors only (verify-anchors.json):
  User Input     the entry function's input parameters (`entry.inputs`) and the calls that read a human's answer
  LLM Response   the LLM calls
Tool calls are not given: they are found between two LLM calls.

The code is read as a data-flow graph (assignments, arguments -> parameters, return values, self.<field>, containers,
loops, comprehensions). An anchored call is a boundary: its arguments leave the agent, what it returns is a new value.
Other calls the code does not show the inside of are tool candidates; until they are chosen, values pass through them:
  external call   an imported library (subprocess.run, requests.get, ...) or a builtin with an effect (exec, input, ...)
  unseen call     a method of an object stored on self whose type is not known (self.env.execute), or a function
                  looked up by key (tool = tools[name]). Decided by where the callee comes from, never by its name.
A tool is a candidate that runs in the agent loop (a loop that also calls the LLM, directly or through the functions
it calls), whose arguments come from an LLM reply (forward from it) and whose result reaches the next LLM call or the
final output (backward from those), that is not a method of the LLM's own object, and that comes first after the
reply (not downstream of another candidate). Its result then becomes a new value too.

A log call (the anchors' logger module) records a value only when it receives the value itself: a name,
self.<field>, or a list/tuple/dict of those. A field, slice or derived value (`[c["name"] for c in reply]`,
`response.stop_reason`, `out[:200]`) records only part of it. Around a tool only the tool's own logs count: on the way
in a value that goes to tools only (not the whole reply, logged by llm_response), on the way out a value that comes
from tool results only (not the message list, logged by llm_query).

Verdict per (value, use), over every path between them:
  검증 완료  every path passes a value that a log call records whole
  일부만 기록  some path does not, and only parts of the value reach a log on it
  누락        some path does not, and no log call touches the value

usage (repo root):
  uv run observatory/flow_verify.py PoC/verification/toy_agent \\
      PoC/verification/toy_agent/observatory/diagram/verify-anchors.json PoC/verification/out_verify \\
      --only agent.py llm.py tools.py
outputs: report.html, report.json, graph.json (the graph per function, with its log calls: the base of the diagram)
"""

import argparse
import ast
import builtins
import html
import json
from collections import deque
from pathlib import Path

KIND = {
    "user_input": "User Input",
    "llm_response": "LLM Response",
    "tool_response": "Tool Response",
    "other": "기타 외부 응답",
}
USE = {"llm_response": "LLM 입력", "user_input": "사용자에게 표시", "tool_response": "Tool 입력", "other": "외부 호출"}
STATUS = {"covered": "검증 완료", "partial": "일부만 기록", "bypass": "누락"}
# builtins that only compute a value from their arguments (no effect outside the process)
PURE = {
    "str", "int", "float", "complex", "bool", "list", "dict", "tuple", "set", "frozenset", "bytes", "bytearray",
    "sorted", "reversed", "enumerate", "zip", "range", "min", "max", "sum", "any", "all", "abs", "round", "repr",
    "format", "iter", "next", "map", "filter", "chr", "ord", "hex", "bin", "oct", "divmod", "pow", "slice", "getattr",
    "super", "vars", "dir", "ascii",
}  # fmt: skip
NO_DATA = {"len", "type", "isinstance", "issubclass", "id", "bool", "hash", "callable", "hasattr"}
MUTATORS = {"append", "extend", "insert", "update", "add", "setdefault", "appendleft", "put"}


class Verifier:
    def __init__(self, root: Path, files: list[Path], anchors: dict):
        self.root, self.anchors, self.logger = root, anchors, anchors.get("logger", "observatory_log")
        self.trees = {f.relative_to(root).as_posix(): ast.parse(f.read_text(encoding="utf-8")) for f in files}
        self.modules = {self.module_id(f) for f in self.trees}
        self.funcs, self.classes, self.module_funcs, self.imports = {}, {}, {}, {}
        self.parent, self.own_names = {}, {}  # nested function -> enclosing function, names local to it
        for file, tree in self.trees.items():
            self.scan(file, tree)
        self.edges, self.info, self.partial = set(), {}, set()
        self.sources, self.sinks, self.logs = set(), set(), set()
        self.var_types, self.field_types = {}, {}
        self.passed = set()  # (caller node, parameter) when the call passes the caller's value itself
        self.mutated = set()  # nodes whose object is changed in place: x.field = v, x[k] = v, x.append(v)
        self.found = set()  # matched source anchors
        self.ext_ret = {}  # boundary call node -> node of the value it returns
        self.calls = {}  # function -> project functions it calls
        self.containers = set()  # (function, variable) holding a dict/list/set a call may fill in place
        self.filled = {}  # boundary call node -> the container variables it fills
        self.through = {}  # candidate call node -> nodes its result is computed from, while it is not a tool
        self.picked = set()  # (function, variable) holding a value looked up by key: tool = tools[name]
        self.return_lines = {}  # (function, node returned) -> line of that return statement
        self.loop_calls = {}  # loop (file, line) -> project functions called in its body
        for q, f in self.funcs.items():  # types of self.<field> first: calls through fields resolve with them
            self.enter(q)
            for s in ast.walk(f["node"]):
                if isinstance(s, ast.Assign):
                    for t in s.targets:
                        if self.is_self_attr(t) and (typ := self.type_of(s.value)):
                            self.field_types[(f["cls"], t.attr)] = typ
        self.entry()
        for q in self.funcs:
            self.enter(q)
            self.stmts(self.funcs[q]["node"].body)
        for n, param in self.passed:  # an object changed through a parameter is the caller's object
            if param in self.mutated:
                self.edge({param}, n)
        if missing := [a for a in anchors.get("sources", []) if json.dumps(a) not in self.found]:
            raise SystemExit(f"anchors not found in the code: {missing}")

    # ---------------------------------------------------------------- project structure
    @staticmethod
    def module_id(file: str) -> str:
        """A project module by its path: `pkg/agent/views` for pkg/agent/views.py, `pkg` for pkg/__init__.py."""
        return Path(file).with_suffix("").as_posix().removesuffix("/__init__")

    def resolve(self, file: str, module: str, level: int) -> str | None:
        """The analyzed module an import names (`.memory` from `file`, or `pkg.agent.views`), if any."""
        base = Path(file).parent.parts[: len(Path(file).parent.parts) - level + 1] if level else ()
        target = "/".join([*base, *filter(None, module.split("."))])
        return next((m for m in self.modules if m == target or (not level and m.endswith(f"/{target}"))), None)

    def scan(self, file: str, tree: ast.Module) -> None:
        module, imports = self.module_id(file), {}
        for n in ast.walk(tree):
            if isinstance(n, ast.Import):
                for a in n.names:
                    found = self.resolve(file, a.name, 0) if a.asname or "." not in a.name else None
                    kind = "logger" if a.name.split(".")[-1] == self.logger else "module" if found else "ext"
                    imports[a.asname or a.name.split(".")[0]] = (kind, found or a.name)
            elif isinstance(n, ast.ImportFrom):
                for a in n.names:
                    full = f"{n.module}.{a.name}" if n.module else a.name
                    if self.logger in ((n.module or "").split(".")[-1], a.name):
                        imports[a.asname or a.name] = ("logger", a.name)
                    elif sub := self.resolve(file, full, n.level):
                        imports[a.asname or a.name] = ("module", sub)
                    elif target := self.resolve(file, n.module or "", n.level):
                        imports[a.asname or a.name] = ("project", f"{target}.{a.name}")
                    else:
                        imports[a.asname or a.name] = ("ext", full)
        self.imports[file] = imports
        for node in tree.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                q = f"{module}.{node.name}"
                self.funcs[q] = {"node": node, "file": file, "cls": None}
                self.module_funcs[(module, node.name)] = q
                self.nested(q)
            elif isinstance(node, ast.ClassDef):
                methods = {}
                for m in node.body:
                    if isinstance(m, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        methods[m.name] = f"{node.name}.{m.name}"
                        self.funcs[methods[m.name]] = {"node": m, "file": file, "cls": node.name}
                        self.nested(methods[m.name])
                bases = [ast.unparse(b).split(".")[-1] for b in node.bases]
                self.classes[node.name] = {"module": module, "methods": methods, "bases": bases}

    def nested(self, q: str) -> None:
        """Functions defined inside q: registered as q.<name>; names they do not bind are q's (a closure)."""
        todo = list(self.funcs[q]["node"].body)
        while todo:
            n = todo.pop()
            if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)):
                inner = f"{q}.{n.name}"
                self.funcs[inner] = self.funcs[q] | {"node": n}
                self.parent[inner] = q
                a = n.args
                self.own_names[inner] = {
                    p.arg for p in [*a.posonlyargs, *a.args, *a.kwonlyargs, a.vararg, a.kwarg] if p
                }
                self.own_names[inner] |= {
                    x.id for x in ast.walk(n) if isinstance(x, ast.Name) and isinstance(x.ctx, ast.Store)
                }
                self.nested(inner)
            elif not isinstance(n, (ast.ClassDef, ast.Lambda)):
                todo += [c for c in ast.iter_child_nodes(n) if isinstance(c, ast.stmt)]

    def mro(self, cls: str) -> list[str]:
        out = [cls]
        for b in self.classes.get(cls, {}).get("bases", []):
            if b in self.classes:
                out += [c for c in self.mro(b) if c not in out]
        return out

    def method(self, cls: str, name: str, subclasses: bool = False) -> list[str]:
        found = next(
            (self.classes[c]["methods"][name] for c in self.mro(cls) if name in self.classes[c]["methods"]), None
        )
        subs = [
            self.classes[s]["methods"][name]
            for s in self.classes
            if subclasses and s != cls and cls in self.mro(s) and name in self.classes[s]["methods"]
        ]
        return [x for x in [found, *subs] if x]

    def global_name(self, name: str):
        """What a module-level name refers to: ("funcs", [q]), ("class", C), ("ext", dotted), ("logger",)."""
        module = self.module_id(self.file)
        if (module, name) in self.module_funcs:
            return ("funcs", [self.module_funcs[(module, name)]])
        if name in self.classes and self.classes[name]["module"] == module:
            return ("class", name)
        kind, dotted = self.imports[self.file].get(name, (None, None))
        if kind == "project":
            mod, attr = dotted.rsplit(".", 1)
            if (mod, attr) in self.module_funcs:
                return ("funcs", [self.module_funcs[(mod, attr)]])
            if attr in self.classes:
                return ("class", attr)
        if kind == "module":
            return ("module", dotted)
        if kind in ("ext", "logger"):
            return (kind, dotted)
        return None

    # ---------------------------------------------------------------- types (only what resolution needs)
    def enter(self, q: str) -> None:
        self.q, self.cls, self.file = q, self.funcs[q]["cls"], self.funcs[q]["file"]
        self.loop = ()  # loops around the statement being read

    def is_self_attr(self, e) -> bool:
        return (
            isinstance(e, ast.Attribute) and isinstance(e.value, ast.Name) and e.value.id == "self" and bool(self.cls)
        )

    def type_of(self, e):
        """("class", C) instance, ("ext", dotted) external object, ("funcs", [q]) function(s) or a container of them."""
        if isinstance(e, ast.Call):
            callee = self.callee(e.func)
            if callee[0] == "class":
                return callee
            if callee[0] == "ext":
                return ("ext", callee[1])
            return None
        if isinstance(e, ast.Name):
            if (self.q, e.id) in self.var_types:
                return self.var_types[(self.q, e.id)]
            g = self.global_name(e.id)
            return g if g and g[0] in ("funcs", "ext") else None
        if self.is_self_attr(e):
            return next(
                (self.field_types[(c, e.attr)] for c in self.mro(self.cls) if (c, e.attr) in self.field_types), None
            )
        if isinstance(e, (ast.Dict, ast.List, ast.Tuple, ast.Set)):
            parts = [self.type_of(x) for x in (e.values if isinstance(e, ast.Dict) else e.elts)]
            if parts and all(p and p[0] == "funcs" for p in parts):
                return ("funcs", sorted({q for p in parts for q in p[1]}))
        if isinstance(e, ast.Attribute):
            base = self.type_of(e.value)
            if base and base[0] == "ext":
                return ("ext", f"{base[1]}.{e.attr}")
            if base and base[0] == "module":
                return self.module_attr(base[1], e.attr)
        return None

    def module_attr(self, module: str, attr: str):
        if (module, attr) in self.module_funcs:
            return ("funcs", [self.module_funcs[(module, attr)]])
        return ("class", attr) if attr in self.classes else None

    def callee(self, f):
        """("log",) ("funcs", [q]) ("class", C) ("ext", dotted) ("pure",) ("nodata",) ("mutator", base) ("value", base)"""
        if isinstance(f, ast.Name):
            q = self.q
            while q and f"{q}.{f.id}" not in self.funcs:  # a function defined in this one or an enclosing one
                q = self.parent.get(q)
            if q:
                return ("funcs", [f"{q}.{f.id}"])
            if (self.q, f.id) in self.var_types:
                t = self.var_types[(self.q, f.id)]
                return t if t[0] in ("funcs", "ext") else ("value", None)
            if g := self.global_name(f.id):
                return g if g[0] != "module" else ("value", None)
            if f.id in NO_DATA:
                return ("nodata",)
            if f.id in PURE:
                return ("pure",)
            if hasattr(builtins, f.id):
                return ("ext", f.id)
            return ("value", None)
        if isinstance(f, ast.Attribute):
            if isinstance(f.value, ast.Name) and f.value.id == "self" and self.cls:
                if methods := self.method(self.cls, f.attr, subclasses=True):
                    return ("funcs", methods)
                return ("value", f.value)  # self.executor(code): an object stored on self, called
            if (
                isinstance(f.value, ast.Name)
                and self.global_name(f.value.id)
                and self.global_name(f.value.id)[0] == "logger"
            ):
                return ("log",)
            base = self.type_of(f.value)
            if base and base[0] == "class":
                return ("funcs", self.method(base[1], f.attr, subclasses=True))
            if base and base[0] == "ext":
                return ("ext", f"{base[1]}.{f.attr}")
            if base and base[0] == "module":
                return self.module_attr(base[1], f.attr) or ("value", f.value)
            return ("mutator", f.value) if f.attr in MUTATORS else ("value", f.value)
        if isinstance(f, ast.Subscript):
            base = self.type_of(f.value)
            if base and base[0] == "funcs":
                return base
        return ("value", None)

    # ---------------------------------------------------------------- graph
    def node(self, name: str, kind: str, line: int, q: str | None = None, **extra) -> str:
        """q: the function the node belongs to when it is not the one being read (a callee's parameter or return)."""
        file, func = (self.funcs[q]["file"], q) if q else (self.file, self.q)
        self.info.setdefault(name, {"kind": kind, "file": file, "line": line, "func": func, **extra})
        return name

    def local(self, q: str, var: str, line: int) -> str:
        while q in self.parent and var not in self.own_names[q]:  # a closure reads the enclosing function's name
            q = self.parent[q]
        if q != self.q:  # a callee's parameter, bound at the call site: place it at the callee's def
            return self.node(f"{q}:{var}", "var", self.funcs[q]["node"].lineno, q=q)
        return self.node(f"{q}:{var}", "var", line)

    def ret(self, q: str) -> str:
        return self.node(f"{q}:<return>", "var", self.funcs[q]["node"].lineno, q=q)

    def edge(self, srcs, dst: str) -> None:
        self.edges |= {(s, dst) for s in srcs if s != dst}

    def entry(self) -> None:
        e = self.anchors["entry"]
        q = e["function"] if e["function"] in self.funcs else next(k for k in self.funcs if k.endswith(e["function"]))
        self.entry_q = q
        self.enter(q)
        a = self.funcs[q]["node"].args
        for p in [*a.posonlyargs, *a.args, *a.kwonlyargs]:
            if p.arg in e.get("inputs", []):
                src = self.node(f"[User Input] {p.arg} (entry parameter)", "source", p.lineno, cat="user_input")
                self.sources.add(src)
                self.edge({src}, self.local(q, p.arg, p.lineno))
        out = self.node(f"[최종 출력] return of {q}", "sink", self.funcs[q]["node"].lineno, use="최종 출력")
        self.sinks.add(out)
        self.edge({self.ret(q)}, out)

    def anchor(self, c: ast.Call):
        return next(
            (
                a
                for a in self.anchors.get("sources", [])
                if a["path"] == self.file and a["line"] == c.lineno and a["call"] == ast.unparse(c.func)
            ),
            None,
        )

    # ---------------------------------------------------------------- statements
    def stmts(self, body) -> None:
        for s in body:
            self.stmt(s)

    def stmt(self, s) -> None:
        if isinstance(s, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            return
        if isinstance(s, ast.Assign):
            v = self.reads(s.value)
            for t in s.targets:
                if isinstance(t, ast.Name) and (typ := self.type_of(s.value)):
                    self.var_types[(self.q, t.id)] = typ
                self.container(t, s.value)
                if isinstance(t, ast.Name) and isinstance(s.value, ast.Subscript):
                    self.picked.add((self.q, t.id))
                self.bind(t, v, s.lineno)
        elif isinstance(s, (ast.AugAssign, ast.AnnAssign)) and s.value is not None:
            self.container(s.target, s.value)
            extra = self.reads(s.target) if isinstance(s, ast.AugAssign) else set()
            self.bind(s.target, self.reads(s.value) | extra, s.lineno)
        elif isinstance(s, ast.Return) and s.value is not None:
            for n in self.reads(s.value):
                self.return_lines.setdefault((self.q, n), s.lineno)
                self.edge({n}, self.ret(self.q))
        elif isinstance(s, (ast.For, ast.AsyncFor, ast.While)):
            if isinstance(s, ast.While):
                self.reads(s.test)
            else:
                self.bind(s.target, self.reads(s.iter), s.lineno)
            self.loop += ((self.file, s.lineno),)
            self.stmts(s.body)
            self.loop = self.loop[:-1]
            self.stmts(s.orelse)
        elif isinstance(s, (ast.With, ast.AsyncWith)):
            for it in s.items:
                v = self.reads(it.context_expr)
                if it.optional_vars is not None:
                    self.bind(it.optional_vars, v, s.lineno)
            self.stmts(s.body)
        elif isinstance(s, ast.Expr):
            self.reads(s.value)
        else:
            for f in ("test", "subject", "exc"):
                if isinstance(getattr(s, f, None), ast.expr):
                    self.reads(getattr(s, f))
            for f in ("body", "orelse", "finalbody"):
                self.stmts(getattr(s, f, []) or [])
            for h in getattr(s, "handlers", []) or []:
                self.stmts(h.body)
            for c in getattr(s, "cases", []) or []:
                self.stmts(c.body)

    def container(self, target, value) -> None:
        literal = (ast.Dict, ast.List, ast.Set, ast.DictComp, ast.ListComp, ast.SetComp)
        built = (
            isinstance(value, ast.Call)
            and isinstance(value.func, ast.Name)
            and value.func.id in ("dict", "list", "set")
        )
        if isinstance(target, ast.Name) and (isinstance(value, literal) or built):
            self.containers.add((self.q, target.id))

    def bind(self, target, srcs: set, line: int) -> None:
        if isinstance(target, ast.Name):
            self.edge(srcs, self.local(self.q, target.id, line))
        elif self.is_self_attr(target):
            self.edge(srcs, self.node(f"{self.cls}.{target.attr}", "field", line))
        elif isinstance(target, (ast.Tuple, ast.List)):
            for e in target.elts:
                self.bind(e.value if isinstance(e, ast.Starred) else e, srcs, line)
        elif isinstance(target, (ast.Subscript, ast.Attribute)):
            for base in self.reads(target.value):
                self.edge(srcs, base)
                self.mutated.add(base)

    # ---------------------------------------------------------------- expressions
    def reads(self, e) -> set:
        if e is None or isinstance(e, ast.Lambda):
            return set()
        if isinstance(e, ast.Name):
            return {self.local(self.q, e.id, e.lineno)}
        if self.is_self_attr(e):
            return {self.node(f"{self.cls}.{e.attr}", "field", e.lineno)}
        if isinstance(e, ast.Attribute):
            return self.reads(e.value)
        if isinstance(e, ast.Subscript):
            self.reads(e.slice)
            return self.reads(e.value)
        if isinstance(e, ast.Call):
            return self.call(e)
        if isinstance(e, (ast.ListComp, ast.SetComp, ast.GeneratorExp, ast.DictComp)):
            for g in e.generators:
                self.bind(g.target, self.reads(g.iter), e.lineno)
                for cond in g.ifs:
                    self.reads(cond)
            parts = [e.key, e.value] if isinstance(e, ast.DictComp) else [e.elt]
            return set().union(*(self.reads(p) for p in parts))
        if isinstance(e, ast.Compare):
            for x in [e.left, *e.comparators]:
                self.reads(x)
            return set()  # a comparison yields a bool, not the compared data
        out = set()
        for child in ast.iter_child_nodes(e):
            if isinstance(child, ast.expr):
                out |= self.reads(child)
            elif isinstance(child, ast.keyword):
                out |= self.reads(child.value)
        return out

    def whole(self, e) -> set:
        """The nodes whose whole value `e` passes on (not a piece of it): a name, self.<field>, or a container of those."""
        if isinstance(e, ast.Starred):
            return self.whole(e.value)
        if isinstance(e, ast.Name):
            return {self.local(self.q, e.id, e.lineno)}
        if self.is_self_attr(e):
            return {self.node(f"{self.cls}.{e.attr}", "field", e.lineno)}
        if isinstance(e, (ast.List, ast.Tuple, ast.Set, ast.Dict)):
            return set().union(*(self.whole(x) for x in (e.values if isinstance(e, ast.Dict) else e.elts)))
        return set()

    def call(self, c: ast.Call) -> set:
        pos = [self.reads(a.value if isinstance(a, ast.Starred) else a) for a in c.args]
        kws = {k.arg: self.reads(k.value) for k in c.keywords}
        every = set().union(*pos, *kws.values())
        loc = f"{self.file}:{c.lineno}"
        kind = self.callee(c.func)
        anchor = self.anchor(c)
        if kind[0] == "log":
            log = self.node(f"[LOG] {ast.unparse(c.func)} @ {loc}", "log", c.lineno)
            self.logs.add(log)
            self.edge(every, log)
            whole = set().union(*(self.whole(a) for a in [*c.args, *(k.value for k in c.keywords)]))
            self.partial |= {(n, log) for n in every - whole}
            return set()
        if anchor:
            self.found.add(json.dumps(anchor))
        if kind[0] in ("funcs", "class") and not anchor:
            targets = kind[1] if kind[0] == "funcs" else self.method(kind[1], "__init__")
            for t in targets:
                self.bind_args(t, c, pos, kws)
                self.calls.setdefault(self.q, set()).add(t)
                for loop in self.loop:
                    self.loop_calls.setdefault(loop, set()).add(t)
            # a new object carries what it was built from (obj.field reads the object; dataclasses have no __init__)
            result = {self.ret(t) for t in targets} if kind[0] == "funcs" else every
        elif kind[0] == "ext" and not anchor and any(self.coroutine(a) for a in c.args):
            result = every  # asyncio.wait_for(self.step(), ...): runs the coroutine and hands back its result
        elif kind[0] == "ext" or anchor or self.unseen(c.func, kind):
            # anchored: a boundary, what comes back is a new value. Otherwise a tool candidate: the value passes
            # through until verdicts() decides it is a tool.
            call, cat = ast.unparse(c.func), anchor["kind"] if anchor else None
            recv = ast.unparse(c.func.value) if isinstance(c.func, ast.Attribute) else None
            sink = self.node(
                f"[외부 호출] {call} @ {loc}", "sink", c.lineno, use=None, cat=cat, recv=recv, loops=self.loop
            )
            self.sinks.add(sink)
            self.edge(every, sink)
            back = self.node(f"[외부 응답] {call} @ {loc}", "source", c.lineno, cat=cat)
            self.ext_ret[sink] = back
            if not anchor:
                self.through[sink] = every | (self.reads(c.func.value) if isinstance(c.func, ast.Attribute) else set())
                self.edge(self.through[sink], back)
            for a in [*c.args, *(k.value for k in c.keywords)]:  # exec(code, scope) fills `scope` in place
                if isinstance(a, ast.Name) and (self.q, a.id) in self.containers:
                    self.edge({back}, self.local(self.q, a.id, c.lineno))
                    self.filled.setdefault(sink, set()).add(self.local(self.q, a.id, c.lineno))
            result = {back}
        elif kind[0] == "nodata":
            result = set()
        elif kind[0] == "mutator":
            for base in self.reads(kind[1]):
                self.edge(every, base)
                self.mutated.add(base)
            result = set()
        else:
            result = every | (self.reads(kind[1]) if kind[0] == "value" and kind[1] is not None else set())
        return result

    def coroutine(self, e) -> bool:
        """A call to an anchored or project async function that is not awaited: a coroutine handed to another call."""
        if not isinstance(e, ast.Call):
            return False
        kind = self.callee(e.func)
        targets = kind[1] if kind[0] == "funcs" else []
        return bool(self.anchor(e)) or (
            bool(targets) and all(isinstance(self.funcs[t]["node"], ast.AsyncFunctionDef) for t in targets)
        )

    def unseen(self, f, kind) -> bool:
        """A call whose implementation the code does not show: a method of an object stored on self whose type is
        not known (a model or environment passed to the constructor), or a function looked up by key (tools[name])."""
        if kind[0] != "value":
            return False
        if isinstance(f, ast.Name):
            return (self.q, f.id) in self.picked
        if not isinstance(f, ast.Attribute):
            return False
        root = f if self.is_self_attr(f) else f.value  # self.executor(code) or self.env.execute(action)
        while isinstance(root, ast.Attribute) and not self.is_self_attr(root):
            root = root.value
        return self.is_self_attr(root) and not self.data_field(root.attr)

    def data_field(self, attr: str) -> bool:
        """self.<attr> holds plain data (a list, dict, string ... built in the class), not an object."""
        data = (ast.List, ast.Dict, ast.Set, ast.Constant, ast.JoinedStr, ast.ListComp, ast.DictComp, ast.SetComp)
        for c in self.mro(self.cls):
            for m in self.classes[c]["methods"].values():
                for s in ast.walk(self.funcs[m]["node"]):
                    targets = (
                        s.targets if isinstance(s, ast.Assign) else [s.target] if isinstance(s, ast.AnnAssign) else []
                    )
                    if any(isinstance(t, ast.Attribute) and t.attr == attr for t in targets) and isinstance(
                        s.value, data
                    ):
                        return True
        return False

    def find_tools(self) -> set[str]:
        """Tool calls: between two LLM calls. They run in the agent loop (a loop that also calls the LLM, directly or
        through the functions it calls), their arguments come from an LLM reply (forward from it) and what they
        return reaches the next LLM call or the final output (backward from those). Methods of the LLM's own object
        are excluded, and of a chain of such calls only the first after the reply is kept."""
        fwd, back = {}, {}
        for a, b in self.edges:
            fwd.setdefault(a, set()).add(b)
            back.setdefault(b, set()).add(a)

        def reach(starts, adj, avoid=frozenset()):
            seen, todo = set(starts), [s for s in starts if s not in avoid]
            while todo:
                for m in adj.get(todo.pop(), ()):
                    if m not in seen and m not in avoid:
                        seen.add(m)
                        todo.append(m)
            return seen

        llm = {k for k in self.ext_ret if self.info[k]["cat"] == "llm_response"}
        replies = {self.ext_ret[k] for k in llm}
        ahead = reach(replies, fwd)
        behind = reach(llm | {k for k in self.sinks if self.info[k]["use"] == "최종 출력"}, back)
        own = {self.info[k]["recv"] for k in llm}

        def region(loop):  # functions run by one pass of the loop
            funcs, todo = set(), list(self.loop_calls.get(loop, ()))
            while todo:
                if (q := todo.pop()) not in funcs:
                    funcs.add(q)
                    todo += self.calls.get(q, ())
            return funcs

        def in_loop(k, loop, funcs):
            return loop in self.info[k]["loops"] or self.info[k]["func"] in funcs

        loops = set(self.loop_calls) | {loop for k in self.ext_ret for loop in self.info[k]["loops"]}
        agent = [(loop, f) for loop in loops for f in [region(loop)] if any(in_loop(k, loop, f) for k in llm)]
        cands = {
            k
            for k, r in self.ext_ret.items()
            if not self.info[k]["cat"]
            and k in ahead
            and r in behind
            and self.info[k]["recv"] not in own
            and any(in_loop(k, loop, f) for loop, f in agent)
        }
        rets = {self.ext_ret[k] for k in cands}
        return {k for k in cands if k in reach(replies, fwd, rets)}

    def bind_args(self, t: str, c: ast.Call, pos: list, kws: dict) -> None:
        a = self.funcs[t]["node"].args
        params = [p.arg for p in [*a.posonlyargs, *a.args]]
        if self.funcs[t]["cls"] and params and params[0] in ("self", "cls"):
            params = params[1:]
        names = params + [p.arg for p in a.kwonlyargs]
        given = [(params[i], arg) for i, arg in enumerate(c.args[: len(params)])]
        given += [(k.arg, k.value) for k in c.keywords if k.arg in names]
        self.passed |= {(n, self.local(t, p, c.lineno)) for p, arg in given for n in self.whole(arg)}
        for i, r in enumerate(pos):
            if i < len(params):
                self.edge(r, self.local(t, params[i], c.lineno))
            elif a.vararg:
                self.edge(r, self.local(t, a.vararg.arg, c.lineno))
        for k, r in kws.items():
            if k is None:  # **kwargs at the call site: any parameter may receive it
                for n in names:
                    self.edge(r, self.local(t, n, c.lineno))
            elif k in names:
                self.edge(r, self.local(t, k, c.lineno))
            elif a.kwarg:
                self.edge(r, self.local(t, a.kwarg.arg, c.lineno))

    # ---------------------------------------------------------------- verdicts
    def verdicts(self) -> list[dict]:
        tools = self.find_tools()
        for k in tools:  # a tool's result is a new value: stop passing the arguments through it
            self.info[k] |= {"cat": "tool_response"}
            self.info[self.ext_ret[k]]["cat"] = "tool_response"
            self.edges -= {(a, self.ext_ret[k]) for a in self.through[k]}
        fwd, back = {}, {}
        for a, b in self.edges:
            fwd.setdefault(a, set()).add(b)
            back.setdefault(b, set()).add(a)
        recorded = {a for a, b in self.edges if b in self.logs and (a, b) not in self.partial}
        while new := {n for n, param in self.passed if param in recorded} - recorded:  # logged by a helper function
            recorded |= new

        def reach(start, adj, avoid=frozenset()):
            seen, todo = {start}, [] if start in avoid else [start]
            while todo:
                for m in adj.get(todo.pop(), ()):
                    if m not in seen and m not in avoid:
                        seen.add(m)
                        todo.append(m)
            return seen

        def path(a, b, avoid=frozenset()):
            prev, dq = {a: None}, deque([] if a in avoid else [a])
            while dq:
                n = dq.popleft()
                if n == b:
                    out = []
                    while n is not None:
                        out.append(n)
                        n = prev[n]
                    return out[::-1]
                for m in fwd.get(n, ()):
                    if m not in prev and m not in avoid:
                        prev[m] = n
                        dq.append(m)
            return []

        def step(n):
            logs = sorted((m for m in fwd.get(n, ()) if m in self.logs), key=lambda m: self.info[m]["line"])
            return self.at(n) | {
                "name": n,
                "kind": self.info[n]["kind"],
                "recorded": n in recorded,
                "logs": [self.at(m) | {"log": m, "whole": (n, m) not in self.partial} for m in logs],
            }

        # starting points: the user input, the anchored calls, and the tools found between LLM calls
        boundary = {k for k in self.ext_ret if self.info[k]["cat"]}
        found = {s: "기준점 (진입점 인자)" for s in sorted(self.sources)}
        found |= {self.ext_ret[k]: "기준점" for k in sorted(boundary - tools)}
        found |= {self.ext_ret[k]: "LLM 응답과 다음 LLM 입력 사이에서 찾음" for k in sorted(tools)}
        for k in boundary:
            self.info[k]["use"] = USE[self.info[k]["cat"]]
        targets = boundary | {k for k in self.sinks if self.info[k]["use"] == "최종 출력"}
        reached = {s: targets & reach(s, fwd) for s in found}
        for k, r in self.ext_ret.items():  # what a call fills in place does not go back into that same call
            if r in reached and k in reached[r] and k not in reach(r, fwd, self.filled.get(k, set())):
                reached[r].discard(k)

        # Around a tool only the tool's own logs count: on the way in, a value recorded whole that goes to tools only
        # (not the whole LLM reply, which also goes back to the LLM); on the way out, a value recorded whole that comes
        # from tool results only (not the message list that mixes them with the task and the replies).
        tool_rets = {self.ext_ret[k] for k in tools}
        into_others = set().union(*(reach(k, back) for k in targets - tools))
        from_others = set().union(*(reach(s, fwd) for s in found if s not in tool_rets))

        rows = []
        for s in sorted(found, key=lambda n: (list(KIND).index(self.info[n]["cat"]), n)):
            down = reach(s, fwd)
            for k in sorted(reached[s]):
                on_path = down & reach(k, back)
                own = (on_path - into_others) if k in tools else (on_path - from_others) if s in tool_rets else on_path
                rec = recorded & own
                full = {m for n in on_path & rec for m in fwd.get(n, ()) if m in self.logs}
                pieces = {m for n in own for m in fwd.get(n, ()) if m in self.logs} - full
                mixed = {m for n in on_path & recorded - own for m in fwd.get(n, ()) if m in self.logs}
                gap = k in reach(s, fwd, rec)  # a path on which nothing is recorded whole
                status = "covered" if not gap else "partial" if pieces else "bypass"
                p = path(s, k, rec) if gap else path(s, k)
                steps = [step(n) for n in p]
                if self.info[k]["use"] == "최종 출력" and len(p) >= 3:  # point at the return statement, not the def
                    line = self.return_lines.get((self.info[k]["func"], p[-3]), self.info[k]["line"])
                    uri = steps[-1]["uri"].rsplit(":", 1)[0] + f":{line}"
                    steps[-1] |= {"line": line, "code": self.line(self.info[k]["file"], line), "uri": uri}
                rows.append(
                    {
                        "value": KIND[self.info[s]["cat"]],
                        "found": found[s],
                        "mixed_logs": sorted(short(m) for m in mixed) if gap else [],
                        "source": s,
                        "use": self.info[k]["use"],
                        "sink": k,
                        "sink_at": {x: steps[-1][x] for x in ("file", "line", "code", "uri")},
                        "status": status,
                        "steps": steps,
                    }
                )
        return rows

    def at(self, n: str) -> dict:
        info = self.info[n]
        uri = f"vscode://file/{(self.root / info['file']).resolve().as_posix()}:{info['line']}"
        return {"file": info["file"], "line": info["line"], "code": self.line(info["file"], info["line"]), "uri": uri}

    def line(self, file: str, line: int) -> str:
        return (self.root / file).read_text(encoding="utf-8").splitlines()[line - 1].strip()


def short(n: str) -> str:
    return n.split(" @ ")[0]


def graph(v: Verifier, rows: list[dict]) -> dict:
    """The data-flow graph between the anchors and their uses, one node per function, field or external call.

    A node carries its code location and the log calls in it (event, line, the values it records and whether whole), so
    that a diagram drawn from it shows where each logged event comes from. An edge is a value moving between two nodes.
    """
    fwd, back = {}, {}
    for a, b in v.edges:
        fwd.setdefault(a, set()).add(b)
        back.setdefault(b, set()).add(a)

    def reach(starts, adj):
        seen, todo = set(starts), list(starts)
        while todo:
            for m in adj.get(todo.pop(), ()):
                if m not in seen:
                    seen.add(m)
                    todo.append(m)
        return seen

    on = reach({r["source"] for r in rows}, fwd) & reach({r["sink"] for r in rows}, back)
    nodes = {}

    def function(q: str) -> str:
        f = v.funcs[q]["node"]
        role = "entry" if q == v.entry_q else "function"
        nodes.setdefault(q, {"id": q, "role": role, "label": q, "path": v.funcs[q]["file"], "line": f.lineno,
                             "end_line": f.end_lineno, "logs": [], "gaps": []})  # fmt: skip
        return q

    def group(n: str) -> str:
        info = v.info[n]
        if info["kind"] == "var":
            return function(info["func"])
        if info["kind"] == "field":
            gid, extra = n, {"role": "state", "label": n}
        elif n.startswith("[User Input]"):
            gid, extra = n, {"role": "user_input", "label": n.split("] ")[1].split(" (")[0]}
        elif n.startswith("[최종 출력]"):
            gid, extra = "final output", {"role": "final_output", "label": f"return of {v.entry_q}"}
        else:
            call, loc = n.split("] ", 1)[1].split(" @ ")
            cat = info.get("cat") or v.info.get(f"[외부 호출] {call} @ {loc}", {}).get("cat")
            gid, extra = f"{call} @ {loc}", {"role": cat or "external", "label": call}
        nodes.setdefault(gid, {"id": gid, **extra, "path": info["file"], "line": info["line"], "logs": [], "gaps": []})
        return gid

    edges = {}
    for a, b in v.edges:
        if a in on and b in on and (ga := group(a)) != (gb := group(b)):
            name = a if v.info[a]["kind"] in ("var", "field") else b
            edges.setdefault((ga, gb), set()).add(name.split(":")[-1].split(".")[-1].split(" @ ")[0].strip("<>"))
    for m in sorted(v.logs, key=lambda m: (v.info[m]["file"], v.info[m]["line"])):
        info = v.info[m]
        records = sorted({(a.split(":")[-1].split(".")[-1], (a, m) not in v.partial) for a in back.get(m, ())})
        nodes[function(info["func"])]["logs"].append(
            {
                "event": short(m).split(".")[-1],
                "at": f"{info['file']}:{info['line']}",
                "code": v.line(info["file"], info["line"]),
                "records": [{"value": n, "whole": w} for n, w in records],
            }
        )
    for r in rows:
        if r["status"] != "covered":
            for gid in dict.fromkeys(group(s["name"]) for s in r["steps"]):
                nodes[gid]["gaps"].append(f"{r['value']} → {r['use']}: {STATUS[r['status']]}")
    return {
        "entry": v.entry_q,
        "nodes": list(nodes.values()),
        "edges": [{"from": a, "to": b, "values": sorted(n)} for (a, b), n in sorted(edges.items())],
        "verdicts": {s: sum(r["status"] == k for r in rows) for k, s in STATUS.items()},
    }


CSS = """
:root { --bg: #f7f8fa; --panel: #fff; --text: #1d2330; --muted: #5d6677; --line: #dde1e8;
  --ok: #2e7d32; --ok-bg: #e7f4ea; --part: #b26a00; --part-bg: #fff2df; --miss: #c62828; --miss-bg: #fde8e8;
  --user: #1f8a4c; --llm: #2457c5; --tool: #8a4fd0; --chip: #f1f3f6; }
@media (prefers-color-scheme: dark) { :root:not([data-theme="light"]) { --bg: #14171c; --panel: #1c2027; --text: #e6e9ef;
  --muted: #9aa3b2; --line: #2e343e; --ok: #6cc070; --ok-bg: #1d3322; --part: #e0a040; --part-bg: #3a2c14;
  --miss: #ef6b6b; --miss-bg: #3d1e1e; --user: #5cc98a; --llm: #7aa2ff; --tool: #b48cf0; --chip: #252a33; } }
* { box-sizing: border-box; }
body { margin: 0; background: var(--bg); color: var(--text); font: 14px/1.5 -apple-system, "Segoe UI", "Apple SD Gothic Neo", "Malgun Gothic", sans-serif; }
main { max-width: 1180px; margin: 0 auto; padding: 24px 16px 64px; }
h1 { font-size: 22px; margin: 0 0 4px; } h2 { font-size: 17px; margin: 32px 0 12px; } .muted { color: var(--muted); }
code { font-family: ui-monospace, Menlo, Consolas, monospace; font-size: 12px; }
.tiles { display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 10px; margin: 16px 0; }
.tile { background: var(--panel); border: 1px solid var(--line); border-radius: 10px; padding: 12px 14px; }
.tile b { display: block; font-size: 26px; line-height: 1.2; } .tile.ok b { color: var(--ok); } .tile.partial b { color: var(--part); }
.tile.bypass b { color: var(--miss); }
.card { background: var(--panel); border: 1px solid var(--line); border-left: 5px solid var(--line); border-radius: 10px; padding: 12px 14px; margin-bottom: 12px; }
.card.covered { border-left-color: var(--ok); } .card.partial { border-left-color: var(--part); } .card.bypass { border-left-color: var(--miss); }
.head { display: flex; flex-wrap: wrap; gap: 8px; align-items: center; font-weight: 600; margin-bottom: 10px; }
.badge { font-size: 12px; font-weight: 600; padding: 1px 8px; border-radius: 999px; color: #fff; }
.badge.covered { background: var(--ok); } .badge.partial { background: var(--part); } .badge.bypass { background: var(--miss); }
.chain { display: flex; flex-wrap: wrap; align-items: flex-start; gap: 6px 4px; }
.arrow { color: var(--muted); padding-top: 8px; }
.step { display: flex; flex-direction: column; gap: 4px; max-width: 230px; }
.chip { display: block; background: var(--chip); border: 1.5px solid var(--line); border-radius: 8px; padding: 5px 9px; color: inherit; text-decoration: none; }
.chip:hover { border-color: var(--muted); }
.chip .name { font-weight: 600; font-size: 13px; word-break: break-all; } .chip .loc { font-size: 11px; color: var(--muted); }
.chip.source.user_input { border-color: var(--user); } .chip.source.llm_response { border-color: var(--llm); } .chip.source.tool_response { border-color: var(--tool); }
.chip.recorded { background: var(--ok-bg); border-color: var(--ok); }
.chip.sink.covered { background: var(--ok-bg); border-color: var(--ok); } .chip.sink.partial { background: var(--part-bg); border-color: var(--part); }
.chip.sink.bypass { background: var(--miss-bg); border-color: var(--miss); }
.log { display: block; font-size: 11px; padding: 2px 7px; border-radius: 6px; text-decoration: none; border: 1px dashed; }
.log.whole { color: var(--ok); border-color: var(--ok); } .log.piece { color: var(--part); border-color: var(--part); }
.legend { display: flex; flex-wrap: wrap; gap: 14px; font-size: 12px; color: var(--muted); margin-top: 8px; }
.legend span::before { content: ""; display: inline-block; width: 10px; height: 10px; border-radius: 3px; margin-right: 5px; vertical-align: -1px; border: 1.5px solid; }
.legend .l-ok::before { background: var(--ok-bg); border-color: var(--ok); } .legend .l-part::before { border-style: dashed; border-color: var(--part); }
.legend .l-miss::before { background: var(--miss-bg); border-color: var(--miss); }
@media (max-width: 600px) { .step { max-width: 100%; } .arrow { padding-top: 0; } }
"""


def step_label(n: str) -> str:
    if n.startswith("["):
        return n.split("] ", 1)[1].split(" @ ")[0]
    return n


def report_html(rows: list[dict], anchors_path: Path) -> str:
    e = html.escape
    count = {s: sum(r["status"] == s for r in rows) for s in STATUS}
    cat = {v: k for k, v in KIND.items()}

    def chip(st: dict, r: dict, last: bool) -> str:
        cls = "chip"
        if st["kind"] == "source":
            cls += f" source {cat[r['value']]}"
        elif last:
            cls += f" sink {r['status']}"
        elif st["recorded"]:
            cls += " recorded"
        kind = {"source": f"생김 · {r['value']}", "sink": f"쓰임 · {r['use']}"}.get(st["kind"], "")
        head = f'<div class="loc">{e(kind)}</div>' if kind else ""
        logs = "".join(
            f'<a class="log {"whole" if g["whole"] else "piece"}" href="{e(g["uri"])}" title="{e(g["code"])}">'
            f"{'✓ 기록' if g['whole'] else '△ 일부만'} · {e(step_label(g['log']).split('.')[-1])} "
            f"{e(g['file'])}:{g['line']}</a>"
            for g in st["logs"]
        )
        return (
            f'<div class="step"><a class="{cls}" href="{e(st["uri"])}" title="{e(st["code"])}">{head}'
            f'<div class="name">{e(step_label(st["name"]))}</div><div class="loc">{e(st["file"])}:{st["line"]}</div></a>'
            f"{logs}</div>"
        )

    def card(r: dict) -> str:
        chain = '<span class="arrow">→</span>'.join(
            chip(st, r, i == len(r["steps"]) - 1) for i, st in enumerate(r["steps"])
        )
        return (
            f'<div class="card {r["status"]}"><div class="head"><span class="badge {r["status"]}">{STATUS[r["status"]]}</span>'
            f"<span>{e(step_label(r['source']))} → {e(r['use'])}: <code>{e(step_label(r['sink']))}</code></span>"
            f'<span class="muted" style="font-weight:400;font-size:12px">발견: {e(r["found"])}</span></div>'
            f'<div class="chain">{chain}</div>'
            + (
                f'<div class="muted" style="margin-top:8px;font-size:12px">툴 로그는 없고, 다른 값과 섞여서만 기록됨: '
                f"{e(', '.join(m.removeprefix('[LOG] ') for m in r['mixed_logs']))}</div>"
                if r["mixed_logs"]
                else ""
            )
            + "</div>"
        )

    tiles = (
        f'<div class="tile"><span class="muted">흐름</span><b>{len(rows)}</b></div>'
        f'<div class="tile ok"><span class="muted">검증 완료</span><b>{count["covered"]}</b></div>'
        f'<div class="tile partial"><span class="muted">일부만 기록</span><b>{count["partial"]}</b></div>'
        f'<div class="tile bypass"><span class="muted">누락</span><b>{count["bypass"]}</b></div>'
    )
    order = list(STATUS)
    sections = "".join(
        f"<h2>{e(value)}</h2>"
        + "".join(card(r) for r in sorted(mine, key=lambda r: (order.index(r["status"]), r["sink"])))
        for value in KIND.values()
        if (mine := [r for r in rows if r["value"] == value])
    )
    return f"""<!doctype html>
<html lang="ko"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Wrapper Verification Report</title><style>{CSS}</style></head><body><main>
<h1>래퍼 검증 리포트</h1>
<div class="muted">기준점: <code>{e(str(anchors_path))}</code> · 칸을 누르면 그 코드 줄이 열립니다</div>
<div class="tiles">{tiles}</div>
<div class="legend"><span class="l-ok">값을 통째로 기록</span><span class="l-part">값의 일부만 기록</span><span class="l-miss">기록 없이 쓰임</span></div>
{sections}
</main></body></html>
"""


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("src_root", type=Path)
    p.add_argument("anchors", type=Path)
    p.add_argument("out_dir", type=Path)
    p.add_argument("--only", nargs="*", default=[])
    a = p.parse_args()
    anchors = json.loads(a.anchors.read_text(encoding="utf-8"))
    files = [a.src_root / f for f in a.only] if a.only else sorted(a.src_root.rglob("*.py"))
    v = Verifier(a.src_root, [f for f in files if f.stem != anchors.get("logger", "observatory_log")], anchors)
    rows = v.verdicts()
    a.out_dir.mkdir(parents=True, exist_ok=True)
    (a.out_dir / "report.json").write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")
    (a.out_dir / "report.html").write_text(report_html(rows, a.anchors), encoding="utf-8")
    (a.out_dir / "graph.json").write_text(json.dumps(graph(v, rows), indent=2, ensure_ascii=False), encoding="utf-8")
    for r in rows:
        mixed = (
            f"   (툴 로그는 없고 섞여서만 기록: {', '.join(m.removeprefix('[LOG] ') for m in r['mixed_logs'])})"
            if r["mixed_logs"]
            else ""
        )
        print(f"{STATUS[r['status']]:<6} {r['value']:<13} {short(r['source'])} → {r['use']}: {short(r['sink'])}{mixed}")
    print(f"wrote {a.out_dir / 'report.html'} and report.json")


if __name__ == "__main__":
    main()

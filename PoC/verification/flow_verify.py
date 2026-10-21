# /// script
# requires-python = ">=3.10"
# dependencies = []
# ///
"""Static verification of an agent's logging, driven by the anchors the architecture step wrote (verify-anchors.json).

Starting from the anchors, follow the three values the logging must capture:
  User Input     the entry function's input parameters, and the anchored user-input calls
  LLM Response   the value an anchored LLM call returns
  Tool Response  the value an anchored tool execution returns
through assignments, arguments -> parameters, return values, self.<field>, containers, loops and comprehensions,
to every place where one of them leaves the agent's code:
  external call  a call into code outside the project: an imported library (subprocess.run, client.messages.create,
                 requests.get, ...) or a builtin with an effect (exec, input, print, open, ...). Decided by where the
                 callee comes from (import, the type of self.<field>), never by its name.
  final output   the entry function's return value

A log call (the anchors' logger module) records a value only when it receives the value itself: a name,
self.<field>, or a list/tuple/dict of those. A field, slice or derived value (`[c["name"] for c in reply]`,
`response.stop_reason`, `out[:200]`) records only part of it.

Verdict per (value, use), over every path between them:
  검증 완료  every path passes a value that a log call records whole
  일부만 기록  some path does not, and only parts of the value reach a log on it
  누락        some path does not, and no log call touches the value
An external call outside every anchored wrapper (not in a wrapper's lines, not in a function called from one) is a
blind spot: the architecture step did not see that the value can leave the agent there.

usage (repo root):
  uv run PoC/verification/flow_verify.py PoC/verification/toy_agent \\
      PoC/verification/toy_agent/observatory/diagram/verify-anchors.json PoC/verification/out_verify \\
      --only agent.py llm.py tools.py
outputs: report.md, report.json
"""

import argparse
import ast
import builtins
import html
import json
from collections import deque
from pathlib import Path

KIND = {"user_input": "User Input", "llm_response": "LLM Response", "tool_response": "Tool Response"}
USE = {"llm_response": "LLM 입력", "user_input": "사용자에게 표시", "tool_response": "Tool 입력"}
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
        self.modules = {Path(f).stem for f in self.trees}
        self.funcs, self.classes, self.module_funcs, self.imports = {}, {}, {}, {}
        for file, tree in self.trees.items():
            self.scan(file, tree)
        self.edges, self.info, self.partial = set(), {}, set()
        self.sources, self.sinks, self.logs = set(), set(), set()
        self.calls: dict[str, set] = {}  # function -> {(file, line, target function)}
        self.var_types, self.field_types = {}, {}
        self.found = set()  # matched source anchors
        self.return_lines = {}  # (function, node returned) -> line of that return statement
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
        if missing := [a for a in anchors["sources"] if json.dumps(a) not in self.found]:
            raise SystemExit(f"anchors not found in the code: {missing}")

    # ---------------------------------------------------------------- project structure
    def scan(self, file: str, tree: ast.Module) -> None:
        module, imports = Path(file).stem, {}
        for n in ast.walk(tree):
            if isinstance(n, ast.Import):
                for a in n.names:
                    top = a.name.split(".")[0]
                    kind = "logger" if a.name == self.logger else "module" if a.name in self.modules else "ext"
                    imports[a.asname or top] = (kind, a.name)
            elif isinstance(n, ast.ImportFrom) and n.module:
                for a in n.names:
                    kind = "logger" if n.module == self.logger else "project" if n.module in self.modules else "ext"
                    imports[a.asname or a.name] = (kind, f"{n.module}.{a.name}")
        self.imports[file] = imports
        for node in tree.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                q = f"{module}.{node.name}"
                self.funcs[q] = {"node": node, "file": file, "cls": None}
                self.module_funcs[(module, node.name)] = q
            elif isinstance(node, ast.ClassDef):
                methods = {}
                for m in node.body:
                    if isinstance(m, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        methods[m.name] = f"{node.name}.{m.name}"
                        self.funcs[methods[m.name]] = {"node": m, "file": file, "cls": node.name}
                bases = [ast.unparse(b).split(".")[-1] for b in node.bases]
                self.classes[node.name] = {"module": module, "methods": methods, "bases": bases}

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
        module = Path(self.file).stem
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
                return ("funcs", self.method(self.cls, f.attr, subclasses=True))
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
    def node(self, name: str, kind: str, line: int, **extra) -> str:
        self.info.setdefault(name, {"kind": kind, "file": self.file, "line": line, "func": self.q, **extra})
        return name

    def local(self, q: str, var: str, line: int) -> str:
        return self.node(f"{q}:{var}", "var", line)

    def ret(self, q: str) -> str:
        return self.node(f"{q}:<return>", "var", self.funcs[q]["node"].lineno)

    def edge(self, srcs, dst: str) -> None:
        self.edges |= {(s, dst) for s in srcs if s != dst}

    def entry(self) -> None:
        e = self.anchors["entry"]
        q = e["function"] if e["function"] in self.funcs else next(k for k in self.funcs if k.endswith(e["function"]))
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
                for a in self.anchors["sources"]
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
                self.bind(t, v, s.lineno)
        elif isinstance(s, (ast.AugAssign, ast.AnnAssign)) and s.value is not None:
            extra = self.reads(s.target) if isinstance(s, ast.AugAssign) else set()
            self.bind(s.target, self.reads(s.value) | extra, s.lineno)
        elif isinstance(s, ast.Return) and s.value is not None:
            for n in self.reads(s.value):
                self.return_lines.setdefault((self.q, n), s.lineno)
                self.edge({n}, self.ret(self.q))
        elif isinstance(s, (ast.For, ast.AsyncFor)):
            self.bind(s.target, self.reads(s.iter), s.lineno)
            self.stmts(s.body + s.orelse)
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
        if kind[0] in ("funcs", "class"):
            targets = kind[1] if kind[0] == "funcs" else self.method(kind[1], "__init__")
            for t in targets:
                self.bind_args(t, c, pos, kws)
                self.calls.setdefault(self.q, set()).add((self.file, c.lineno, t))
            result = {self.ret(t) for t in targets} if kind[0] == "funcs" else set()
        elif kind[0] == "ext":
            use = USE[anchor["kind"]] if anchor else "외부 호출"
            sink = self.node(f"[{use}] {ast.unparse(c.func)} @ {loc}", "sink", c.lineno, use=use)
            self.sinks.add(sink)
            self.edge(every, sink)
            result = every | (self.reads(c.func.value) if isinstance(c.func, ast.Attribute) else set())
        elif kind[0] == "nodata":
            result = set()
        elif kind[0] == "mutator":
            for base in self.reads(kind[1]):
                self.edge(every, base)
            result = set()
        else:
            result = every | (self.reads(kind[1]) if kind[0] == "value" and kind[1] is not None else set())
        if anchor:
            self.found.add(json.dumps(anchor))
            src = self.node(
                f"[{KIND[anchor['kind']]}] {anchor['call']} @ {loc}", "source", c.lineno, cat=anchor["kind"]
            )
            self.sources.add(src)
            return {src}
        return result

    def bind_args(self, t: str, c: ast.Call, pos: list, kws: dict) -> None:
        a = self.funcs[t]["node"].args
        params = [p.arg for p in [*a.posonlyargs, *a.args]]
        if self.funcs[t]["cls"] and params and params[0] in ("self", "cls"):
            params = params[1:]
        names = params + [p.arg for p in a.kwonlyargs]
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
    def wrapped(self) -> set[str]:
        """Functions that run inside an anchored wrapper: called from a wrapper's lines, then everything they call."""
        inside = lambda file, line: any(  # noqa: E731
            w["path"] == file and w["line"] <= line <= w["end_line"] for w in self.anchors["wrappers"]
        )
        todo = [t for calls in self.calls.values() for file, line, t in calls if inside(file, line)]
        seen = set()
        while todo:
            q = todo.pop()
            if q not in seen:
                seen.add(q)
                todo += [t for _, _, t in self.calls.get(q, ())]
        return seen

    def in_wrapper(self, sink: str, wrapped: set[str]) -> bool:
        info = self.info[sink]
        if info["use"] == "최종 출력":
            return True
        return info["func"] in wrapped or any(
            w["path"] == info["file"] and w["line"] <= info["line"] <= w["end_line"] for w in self.anchors["wrappers"]
        )

    def verdicts(self) -> list[dict]:
        fwd, back = {}, {}
        for a, b in self.edges:
            fwd.setdefault(a, set()).add(b)
            back.setdefault(b, set()).add(a)
        recorded = {a for a, b in self.edges if b in self.logs and (a, b) not in self.partial}

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

        wrapped, rows = self.wrapped(), []
        for s in sorted(self.sources, key=lambda n: (list(KIND).index(self.info[n]["cat"]), n)):
            down = reach(s, fwd)
            for k in sorted(k for k in self.sinks if k in down):
                on_path = down & reach(k, back)
                full = sorted({m for n in on_path & recorded for m in fwd.get(n, ()) if m in self.logs})
                pieces = sorted({m for n in on_path for m in fwd.get(n, ()) if m in self.logs} - set(full))
                gap = k in reach(s, fwd, recorded)  # a path on which nothing is recorded whole
                status = "covered" if not gap else "partial" if pieces else "bypass"
                p = path(s, k, recorded) if gap else path(s, k)
                steps = [step(n) for n in p]
                if self.info[k]["use"] == "최종 출력" and len(p) >= 3:  # point at the return statement, not the def
                    line = self.return_lines.get((self.info[k]["func"], p[-3]), self.info[k]["line"])
                    uri = steps[-1]["uri"].rsplit(":", 1)[0] + f":{line}"
                    steps[-1] |= {"line": line, "code": self.line(self.info[k]["file"], line), "uri": uri}
                rows.append(
                    {
                        "value": KIND[self.info[s]["cat"]],
                        "source": s,
                        "source_at": self.at(s),
                        "use": self.info[k]["use"],
                        "sink": k,
                        "sink_at": {x: steps[-1][x] for x in ("file", "line", "code", "uri")},
                        "status": status,
                        "logs": [self.at(m) | {"log": m} for m in sorted(full, key=lambda m: self.info[m]["line"])],
                        "partial_logs": [
                            self.at(m) | {"log": m} for m in sorted(pieces, key=lambda m: self.info[m]["line"])
                        ],
                        "path": p,
                        "steps": steps,
                        "blind_spot": not self.in_wrapper(k, wrapped),
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


def report(rows: list[dict], anchors_path: Path) -> str:
    count = {s: sum(r["status"] == s for r in rows) for s in STATUS}
    blind = [r for r in rows if r["blind_spot"]]
    lines = [
        "# 래퍼 검증 리포트",
        "",
        f"기준점: `{anchors_path}`",
        "",
        f"흐름 {len(rows)}개: **검증 완료 {count['covered']}** · **일부만 기록 {count['partial']}** · "
        f"**누락 {count['bypass']}** · 사각지대 {len({r['sink'] for r in blind})}곳",
        "",
        "## 사각지대 (래퍼 밖에서 값이 외부로 나가는 호출)",
        "",
    ]
    if blind:
        lines += ["| 위치 | 호출 | 들어가는 값 |", "|---|---|---|"]
        for sink in sorted({r["sink"] for r in blind}):
            r = next(r for r in blind if r["sink"] == sink)
            values = ", ".join(sorted({x["value"] for x in blind if x["sink"] == sink}))
            lines.append(f"| `{r['sink_at']['file']}:{r['sink_at']['line']}` | `{r['sink_at']['code']}` | {values} |")
    else:
        lines.append("없음. 세 값이 외부로 나가는 호출은 모두 기준점의 래퍼 안에 있습니다.")
    for value in KIND.values():
        mine = [r for r in rows if r["value"] == value]
        if not mine:
            continue
        lines += ["", f"## {value}", ""]
        for r in sorted(mine, key=lambda r: (list(STATUS).index(r["status"]), r["sink"])):
            mark = {"covered": "✅", "partial": "⚠️", "bypass": "❌"}[r["status"]]
            src, use = r["source_at"], r["sink_at"]
            lines.append(f"### {mark} {STATUS[r['status']]}: {short(r['source'])} → {short(r['sink'])}")
            lines.append("")
            lines.append(f"- 생김: `{src['file']}:{src['line']}` `{src['code']}`")
            for log in r["logs"]:
                lines.append(f"- 기록: `{log['file']}:{log['line']}` `{log['code']}`")
            for log in r["partial_logs"]:
                lines.append(f"- 일부만 기록: `{log['file']}:{log['line']}` `{log['code']}`")
            lines.append(f"- 쓰임 ({r['use']}): `{use['file']}:{use['line']}` `{use['code']}`")
            if r["status"] != "covered":
                lines.append(f"- 기록 없이 지나는 경로: {' → '.join(short(n) for n in r['path'])}")
            if r["blind_spot"]:
                lines.append("- **사각지대**: 이 호출은 기준점의 어떤 래퍼 안에도 없습니다.")
            lines.append("")
    return "\n".join(lines)


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
.tile.bypass b, .tile.blind b { color: var(--miss); }
.card { background: var(--panel); border: 1px solid var(--line); border-left: 5px solid var(--line); border-radius: 10px; padding: 12px 14px; margin-bottom: 12px; }
.card.covered { border-left-color: var(--ok); } .card.partial { border-left-color: var(--part); } .card.bypass { border-left-color: var(--miss); }
.head { display: flex; flex-wrap: wrap; gap: 8px; align-items: center; font-weight: 600; margin-bottom: 10px; }
.badge { font-size: 12px; font-weight: 600; padding: 1px 8px; border-radius: 999px; color: #fff; }
.badge.covered { background: var(--ok); } .badge.partial { background: var(--part); } .badge.bypass, .badge.blind { background: var(--miss); }
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
.note { margin-top: 10px; font-size: 13px; color: var(--muted); }
.legend { display: flex; flex-wrap: wrap; gap: 14px; font-size: 12px; color: var(--muted); margin-top: 8px; }
.legend span::before { content: ""; display: inline-block; width: 10px; height: 10px; border-radius: 3px; margin-right: 5px; vertical-align: -1px; border: 1.5px solid; }
.legend .l-ok::before { background: var(--ok-bg); border-color: var(--ok); } .legend .l-part::before { border-style: dashed; border-color: var(--part); }
.legend .l-miss::before { background: var(--miss-bg); border-color: var(--miss); }
table { width: 100%; border-collapse: collapse; background: var(--panel); border: 1px solid var(--line); border-radius: 10px; overflow: hidden; }
th, td { text-align: left; padding: 8px 10px; border-bottom: 1px solid var(--line); vertical-align: top; } th { font-size: 12px; color: var(--muted); }
a.code { color: inherit; }
@media (max-width: 600px) { .step { max-width: 100%; } .arrow { padding-top: 0; } }
"""


def step_label(n: str) -> str:
    if n.startswith("["):
        return n.split("] ", 1)[1].split(" @ ")[0]
    return n


def report_html(rows: list[dict], anchors_path: Path) -> str:
    e = html.escape
    count = {s: sum(r["status"] == s for r in rows) for s in STATUS}
    blind = sorted({r["sink"] for r in rows if r["blind_spot"]})
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
        note = {
            "covered": "모든 경로가 값을 통째로 기록하는 로그를 지납니다. 위는 그 중 가장 짧은 경로입니다.",
            "partial": "위 경로에서는 값의 일부(필드, 슬라이스, 가공값)만 로그에 들어갑니다.",
            "bypass": "위 경로에서는 어떤 로그도 값을 받지 않습니다.",
        }[r["status"]]
        blind_badge = '<span class="badge blind">사각지대</span>' if r["blind_spot"] else ""
        return (
            f'<div class="card {r["status"]}"><div class="head"><span class="badge {r["status"]}">{STATUS[r["status"]]}</span>'
            f"{blind_badge}<span>{e(step_label(r['source']))} → {e(r['use'])}: <code>{e(step_label(r['sink']))}</code></span></div>"
            f'<div class="chain">{chain}</div><div class="note">{note}</div></div>'
        )

    tiles = (
        f'<div class="tile"><span class="muted">흐름</span><b>{len(rows)}</b></div>'
        f'<div class="tile ok"><span class="muted">검증 완료</span><b>{count["covered"]}</b></div>'
        f'<div class="tile partial"><span class="muted">일부만 기록</span><b>{count["partial"]}</b></div>'
        f'<div class="tile bypass"><span class="muted">누락</span><b>{count["bypass"]}</b></div>'
        f'<div class="tile blind"><span class="muted">사각지대</span><b>{len(blind)}</b></div>'
    )
    if blind:
        trs = "".join(
            f'<tr><td><a class="code" href="{e(r["sink_at"]["uri"])}"><code>{e(r["sink_at"]["file"])}:{r["sink_at"]["line"]}</code></a></td>'
            f"<td><code>{e(r['sink_at']['code'])}</code></td>"
            f"<td>{e(', '.join(sorted({x['value'] for x in rows if x['sink'] == k})))}</td></tr>"
            for k in blind
            for r in [next(x for x in rows if x["sink"] == k)]
        )
        blind_html = f"<table><tr><th>위치</th><th>호출</th><th>들어가는 값</th></tr>{trs}</table>"
    else:
        blind_html = '<p class="muted">없음. 세 값이 외부로 나가는 호출은 모두 기준점의 래퍼 안에 있습니다.</p>'
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
<h2>사각지대 <span class="muted" style="font-size:13px;font-weight:400">래퍼 밖에서 값이 외부로 나가는 호출</span></h2>
{blind_html}
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
    (a.out_dir / "report.md").write_text(report(rows, a.anchors), encoding="utf-8")
    (a.out_dir / "report.html").write_text(report_html(rows, a.anchors), encoding="utf-8")
    for r in rows:
        flag = "  [사각지대]" if r["blind_spot"] else ""
        print(f"{STATUS[r['status']]:<6} {r['value']:<13} {short(r['source'])} → {short(r['sink'])}{flag}")
    print(f"wrote {a.out_dir / 'report.md'} and report.html")


if __name__ == "__main__":
    main()

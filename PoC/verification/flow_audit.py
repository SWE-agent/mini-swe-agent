# /// script
# requires-python = ">=3.10"
# dependencies = ["astor", "graphviz", "py2cfg"]
# ///
"""Data-flow view of an agent's I/O: follow User Input, LLM Response and Tool Response values from where they are
created to where they are used, and check that a log records each value on its way.

  source   the value a call returns: an LLM call (R1-R3), a tool execution (T1, T2, T4), a user-input read (I1),
           plus the parameters of the --entry function (the task)
  sink     where such a value is used: an argument of an LLM call (LLM input), of a tool execution (tool input),
           or the return value of the --entry function (final output)
  log      a logging call (io_audit.py's rules: "[LLM_" / "[TOOL_" tags, agent_observatory_kit functions)

The flow is followed through assignments, call arguments -> parameters, return values, `self.<field>`, container
writes (append, d[k] = v), for loops and comprehensions. `self.x = SomeClass(...)` types the field, so
`self.x.method()` resolves to SomeClass.method; other unknown calls pass their arguments' taint to their result.
len/type/isinstance/bool/id/hash results carry no data. Over-approximate on purpose, like the other tools.

Verdict per (source, sink): "logged" when some value on a path from the source to the sink is an argument of a log
call, "not logged" otherwise, "unused" when the source reaches no sink.

usage:
  uv run --python 3.12 PoC/verification/flow_audit.py PoC/verification/toy_agent PoC/verification/out_flow \
      --entry ToyAgent.run --only agent.py llm.py tools.py
outputs: flow.svg, flow_checks.tsv, cfg/, index.html (viewer, "데이터 흐름" tab)
"""

import argparse
import ast
import json
import sys
from collections import deque
from pathlib import Path

import graphviz

sys.path.insert(0, str(Path(__file__).resolve().parent))
import audit  # noqa: E402
import audit_rules  # noqa: E402
import io_audit  # noqa: E402
import io_rules  # noqa: E402

SOURCE = {"LLM": "LLM Response", "TOOL": "Tool Response", "INPUT": "User Input"}
SINK_LLM, SINK_TOOL, SINK_OUT = "LLM 입력", "Tool 입력", "최종 출력"
NO_DATA = {"len", "type", "isinstance", "id", "bool", "hash", "callable", "hasattr"}
MUTATORS = {"append", "extend", "insert", "update", "add", "setdefault", "put", "appendleft"}
COLOR = {"LLM": "#d00000", "TOOL": "#d46b00", "INPUT": "#1f8a4c"}
FILL = {"LLM": "#ffd6d6", "TOOL": "#ffe3c4", "INPUT": "#d3f0de"}
TAB, NL = "\t", "\n"


class Flow:
    def __init__(self, root, funcs, classes, imports, cls_nodes, entries):
        self.root, self.funcs, self.classes, self.imports, self.cls_nodes = root, funcs, classes, imports, cls_nodes
        self.edges = set()  # (src, dst, line)
        self.info = {}  # node -> {"kind", "file", "line", "func", "cat"}
        self.sources, self.sinks, self.logs = set(), set(), set()
        self.field_types = {
            (i["cls"], t.attr): s.value.func.id
            for i in funcs.values()
            if i["cls"]
            for s in ast.walk(i["node"])
            if isinstance(s, ast.Assign) and isinstance(s.value, ast.Call) and isinstance(s.value.func, ast.Name)
            and s.value.func.id in classes
            for t in s.targets
            if isinstance(t, ast.Attribute) and isinstance(t.value, ast.Name) and t.value.id == "self"
        }
        self.registered = {(a, b) for a, b, _, _ in io_audit.registered_edges(self._t4_sites(), funcs)}
        for q in entries:
            self._entry(q)
        for q in funcs:
            self.q, self.cls, self.file = q, funcs[q]["cls"], funcs[q]["file"]
            self.strings = audit_rules.class_strings(self.cls, classes, cls_nodes)
            self.stmts(funcs[q]["node"].body)

    # ---------------------------------------------------------------- nodes
    def node(self, name, kind, line, **extra):
        self.info.setdefault(name, {"kind": kind, "file": self.file, "line": line, "func": self.q, **extra})
        return name

    def local(self, q, var, line):
        return self.node(f"{q} : {var}", "var", line)

    def ret(self, q):
        return self.node(f"{q} : <return>", "var", self.funcs[q]["node"].lineno)

    def field(self, cls, attr, line):
        return self.node(f"{cls}.{attr}", "field", line)

    def edge(self, srcs, dst, line):
        self.edges |= {(s, dst, line) for s in srcs if s != dst}

    def _t4_sites(self):
        sites = []
        for q, i in self.funcs.items():
            for c in i["calls"]:
                hit = io_rules.classify(c, self.imports[i["file"]], {})
                if hit and hit[1] == "T4-dispatch":
                    sites.append({"func": q, "line": c.lineno, "rule": "T4-dispatch"})
        return sites

    def _entry(self, q):
        self.q, self.cls, self.file = q, self.funcs[q]["cls"], self.funcs[q]["file"]
        a = self.funcs[q]["node"].args
        for p in [*a.posonlyargs, *a.args, *a.kwonlyargs][1 if self.cls else 0 :]:
            src = self.node(f"[User Input] {p.arg} (entry parameter) @ {self.file}:{p.lineno}", "source", p.lineno, cat="INPUT")
            self.sources.add(src)
            self.edge({src}, self.local(q, p.arg, p.lineno), p.lineno)
        out = self.node(f"[{SINK_OUT}] return of {q}", "sink", self.funcs[q]["node"].lineno)
        self.sinks.add(out)
        self.edge({self.ret(q)}, out, self.funcs[q]["node"].lineno)

    # ---------------------------------------------------------------- statements
    def stmts(self, body):
        for s in body:
            self.stmt(s)

    def stmt(self, s):
        if isinstance(s, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            return
        if isinstance(s, ast.Assign):
            v = self.reads(s.value)
            for t in s.targets:
                self.bind(t, v, s.lineno)
        elif isinstance(s, (ast.AugAssign, ast.AnnAssign)) and s.value is not None:
            self.bind(s.target, self.reads(s.value) | (self.reads(s.target) if isinstance(s, ast.AugAssign) else set()), s.lineno)
        elif isinstance(s, ast.Return) and s.value is not None:
            self.edge(self.reads(s.value), self.ret(self.q), s.lineno)
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

    def bind(self, target, srcs, line):
        if isinstance(target, ast.Name):
            self.edge(srcs, self.local(self.q, target.id, line), line)
        elif isinstance(target, ast.Attribute) and isinstance(target.value, ast.Name) and target.value.id == "self" and self.cls:
            self.edge(srcs, self.field(self.cls, target.attr, line), line)
        elif isinstance(target, (ast.Tuple, ast.List)):
            for e in target.elts:
                self.bind(e.value if isinstance(e, ast.Starred) else e, srcs, line)
        elif isinstance(target, (ast.Subscript, ast.Attribute)):
            for base in self.reads(target.value):
                self.edge(srcs, base, line)

    # ---------------------------------------------------------------- expressions
    def reads(self, e) -> set:
        if e is None:
            return set()
        if isinstance(e, ast.Name):
            return {self.local(self.q, e.id, e.lineno)}
        if isinstance(e, ast.Attribute):
            if isinstance(e.value, ast.Name) and e.value.id == "self" and self.cls:
                return {self.field(self.cls, e.attr, e.lineno)}
            return self.reads(e.value)
        if isinstance(e, ast.Subscript):
            self.reads(e.slice)
            return self.reads(e.value)
        if isinstance(e, ast.Call):
            return self.call(e)
        if isinstance(e, ast.Lambda):
            return set()
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

    def call(self, c: ast.Call) -> set:
        pos = [self.reads(a.value if isinstance(a, ast.Starred) else a) for a in c.args]
        kws = {k.arg: self.reads(k.value) for k in c.keywords}
        every = set().union(*pos, *kws.values()) if (pos or kws) else set()
        loc = f"{self.file}:{c.lineno}"
        if any(io_audit.logs(c, cat) for cat in io_audit.WRAPPERS):
            log = self.node(f"[LOG] {ast.unparse(c.func)} @ {loc}", "log", c.lineno)
            self.logs.add(log)
            self.edge(every, log, c.lineno)
            return set()
        hit = io_rules.classify(c, self.imports[self.file], self.strings)
        cat = hit[0] if hit and hit[0] in SOURCE and hit[1] != "T3-toolcall" else None
        if cat:
            if cat != "INPUT":  # the call's arguments are the LLM's / the tool's input
                sink = self.node(f"[{SINK_LLM if cat == 'LLM' else SINK_TOOL}] {hit[2]} @ {loc}", "sink", c.lineno)
                self.sinks.add(sink)
                self.edge(every, sink, c.lineno)
            src = self.node(f"[{SOURCE[cat]}] {hit[2]} @ {loc}", "source", c.lineno, cat=cat)
            self.sources.add(src)
            if hit[1] != "T4-dispatch":
                return {src}
        f = c.func
        name = f.attr if isinstance(f, ast.Attribute) else getattr(f, "id", "")
        if isinstance(f, ast.Name) and name in NO_DATA:
            return set()
        if isinstance(f, ast.Attribute) and name in MUTATORS:
            for base in self.reads(f.value):
                self.edge(every, base, c.lineno)
            return set()
        targets = self.resolve(c)
        for t in targets:
            self.bind_args(t, c, pos, kws)
        result = {self.ret(t) for t in targets}
        if cat:  # T4 dispatch: the registered functions' results are the tool response
            src = f"[{SOURCE[cat]}] {hit[2]} @ {loc}"
            self.edge(result, src, c.lineno)
            return {src}
        if targets:
            return result
        return every | (self.reads(f.value) if isinstance(f, ast.Attribute) else set())

    def resolve(self, c):
        f = c.func
        if isinstance(f, ast.Name):
            return [k for k in self.funcs if k.endswith(f"#{f.id}")][:1]
        if isinstance(f, ast.Attribute):
            v = f.value
            if isinstance(v, ast.Name) and v.id == "self" and self.cls:
                t = audit.resolve(self.cls, f.attr, self.classes)
                subs = [f"{s}.{f.attr}" for s in self.classes if s != self.cls and self.cls in audit.mro(s, self.classes)]
                return [x for x in [t, *subs] if x in self.funcs]
            if isinstance(v, ast.Attribute) and isinstance(v.value, ast.Name) and v.value.id == "self" and self.cls:
                typ = next((self.field_types[(k, v.attr)] for k in audit.mro(self.cls, self.classes) if (k, v.attr) in self.field_types), None)
                if typ:
                    t = audit.resolve(typ, f.attr, self.classes)
                    return [t] if t in self.funcs else []
        if isinstance(f, ast.Subscript):
            return sorted(b for a, b in self.registered if a == self.q)
        return []

    def bind_args(self, t, c, pos, kws):
        a = self.funcs[t]["node"].args
        params = [p.arg for p in [*a.posonlyargs, *a.args]]
        if self.funcs[t]["cls"] and params and params[0] in ("self", "cls"):
            params = params[1:]
        names = params + [p.arg for p in a.kwonlyargs]
        line = c.lineno
        for i, r in enumerate(pos):
            if i < len(params):
                self.edge(r, self.local(t, params[i], line), line)
            elif a.vararg:
                self.edge(r, self.local(t, a.vararg.arg, line), line)
        for k, r in kws.items():
            if k is None:  # **kwargs at the call site: any parameter may receive it
                for n in names:
                    self.edge(r, self.local(t, n, line), line)
            elif k in names:
                self.edge(r, self.local(t, k, line), line)
            elif a.kwarg:
                self.edge(r, self.local(t, a.kwarg.arg, line), line)

    # ---------------------------------------------------------------- verdicts
    def verdicts(self):
        fwd, back = {}, {}
        for a, b, _ in self.edges:
            fwd.setdefault(a, set()).add(b)
            back.setdefault(b, set()).add(a)

        def reach(start, adj):
            seen, todo = {start}, [start]
            while todo:
                for m in adj.get(todo.pop(), ()):
                    if m not in seen:
                        seen.add(m)
                        todo.append(m)
            return seen

        def path(a, b):
            prev, dq = {a: None}, deque([a])
            while dq:
                n = dq.popleft()
                if n == b:
                    out = []
                    while n is not None:
                        out.append(n)
                        n = prev[n]
                    return out[::-1]
                for m in fwd.get(n, ()):
                    if m not in prev:
                        prev[m] = n
                        dq.append(m)
            return []

        rows = []
        for s in sorted(self.sources):
            down = reach(s, fwd)
            sinks = sorted(k for k in self.sinks if k in down)
            for k in sinks or [None]:
                on_path = down & reach(k, back) if k else down
                logged = sorted({m for n in on_path for m in fwd.get(n, ()) if m in self.logs})
                rows.append({
                    "source": s,
                    "sink": k,
                    "status": "outside" if k is None else ("covered" if logged else "bypass"),
                    "path": path(s, k) if k else [],
                    "via": logged,
                    "nodes": sorted(on_path | set(logged)) if k else [],  # everything this flow touches (viewer)
                })
        return rows


def draw(flow, out: Path, rows):
    keep = set()
    for r in rows:
        keep |= set(r["path"]) | set(r["via"]) | {r["source"]}
    for r in rows:  # the log calls hang off the path nodes
        for a, b, _ in flow.edges:
            if b in r["via"] and a in keep:
                keep.add(a)
    font = "Malgun Gothic"  # node labels hold Korean sink names; Graphviz's default Times has no Hangul
    g = graphviz.Digraph(
        "flow",
        graph_attr={"rankdir": "LR", "fontname": font},
        node_attr={"fontsize": "10", "fontname": font},
        edge_attr={"fontname": font},
    )
    ids = {n: f"n{i}" for i, n in enumerate(sorted(keep))}
    for n, i in ids.items():
        info, kind = flow.info[n], flow.info[n]["kind"]
        if kind == "source":
            cat = info["cat"]
            g.node(i, n.replace(" @ ", "\n"), shape="ellipse", style="filled", fillcolor=FILL[cat], color=COLOR[cat], penwidth="2")
        elif kind == "sink":
            g.node(i, n.replace(" @ ", "\n"), shape="doubleoctagon", color="#333333")
        elif kind == "log":
            g.node(i, n.replace(" @ ", "\n"), shape="note", style="filled", fillcolor="#e6f4ea", color="#2e7d32")
        else:
            func, _, var = n.rpartition(" : ")
            label = f"{var}\n{func}" if func else n
            g.node(i, label.replace("\\", "\\\\"), shape="box", style="rounded" + (",bold" if kind == "field" else ""))
    for a, b, line in sorted(flow.edges):
        if a in ids and b in ids:
            log = flow.info[b]["kind"] == "log"
            g.edge(ids[a], ids[b], label=f"L{line}", fontsize="8", **({"color": "#2e7d32", "style": "dashed"} if log else {}))
    g.render(out / "flow", format="svg", cleanup=True)
    return {"svg": (out / "flow.svg").read_text(encoding="utf-8"), "ids": {i: n for n, i in ids.items()}}


LEGEND = {
    "categories": [{"color": COLOR[c], "fill": FILL[c], "label": SOURCE[c]} for c in SOURCE],
    "shapes": [
        {"style": "background:#eee;border-color:#666;border-radius:50%", "label": "출발점 (source): 값이 처음 생기는 호출"},
        {"style": "border-color:#666;border-radius:6px", "label": "값이 지나가는 변수 · 매개변수 · 반환값"},
        {"style": "border-color:#000;border-width:3px;border-radius:6px", "label": "self.필드 (굵은 테두리)"},
        {"style": "border-color:#333;border-style:double;border-width:4px", "label": "사용 지점 (sink): LLM 입력 · Tool 입력 · 최종 출력"},
        {"style": "background:#e6f4ea;border-color:#2e7d32", "label": "로그 호출 (값을 인자로 받음)"},
    ],
    "edges": [
        {"color": "#000000", "style": "solid", "label": "데이터 흐름 (숫자는 줄 번호)"},
        {"color": "#2e7d32", "style": "dashed", "label": "값이 로그로 들어감"},
    ],
}


def write_viewer(flow, out: Path, graph, rows):
    nodes, rel = {}, {}
    for n, info in flow.info.items():
        uri = f"vscode://file/{(flow.root / info['file']).resolve().as_posix()}:{info['line']}"
        cfg = out / "cfg" / f"{audit.cfg_name(info['func'])}.svg"
        nodes[n] = {"file": info["file"], "line": info["line"], "end": info["line"], "uri": uri,
                    "cfg": cfg.relative_to(out).as_posix() if cfg.exists() else None, "calls": []}
    for a, b, line in sorted(flow.edges):
        uri = nodes[a]["uri"].rsplit(":", 1)[0] + f":{line}"
        rel.setdefault(a, {"in": [], "out": []})["out"].append({"name": b, "line": line, "kind": "flow"})
        rel.setdefault(b, {"in": [], "out": []})["in"].append({"name": a, "line": line, "kind": "flow", "file": nodes[a]["file"], "uri": uri})
    def at(name, path=()):
        """Where a node is in the code. The final output points at the entry's return statement that mentions a
        variable or function of the flow's path (an entry can have several returns)."""
        info = flow.info[name]
        line = info["line"]
        if name.startswith(f"[{SINK_OUT}]"):
            returns = [r for r in ast.walk(flow.funcs[info["func"]]["node"]) if isinstance(r, ast.Return) and r.value]
            words = {w for n in path for w in (n.rpartition(" : ")[2], n.rpartition(" : ")[0].split(".")[-1]) if w and w != "<return>"}
            line = next((r.lineno for r in returns if any(w in ast.unparse(r.value) for w in words)), returns[0].lineno)
        return {"file": info["file"], "line": line}

    checks, code = [], {}
    for r in rows:
        info = flow.info[r["source"]]
        checks.append({
            "id": f"{r['source']} -> {r['sink']}", "cat": info["cat"], "rule": r["sink"] or "사용 지점 없음",
            "pattern": r["source"].split("] ", 1)[1].split(" @ ")[0], "detail": "", "func": r["source"],
            "file": info["file"], "line": info["line"], "uri": nodes[r["source"]]["uri"],
            "status": r["status"], "path": r["path"], "via": r["via"],
            # the three questions: where the value is created, where a log records it, where it is used
            "src_at": at(r["source"]), "logs_at": [at(v) for v in r["via"]] if r["sink"] else [],
            "use_at": at(r["sink"], r["path"]) if r["sink"] else None,
            "use_kind": r["sink"].split("]", 1)[0].lstrip("[") if r["sink"] else "",
            "nodes": r["nodes"],
        })
        for loc in [checks[-1]["src_at"], *checks[-1]["logs_at"], checks[-1]["use_at"]]:
            if loc:
                lines = (flow.root / loc["file"]).read_text(encoding="utf-8").splitlines()
                code[f"{loc['file']}:{loc['line']}"] = lines[loc["line"] - 1].strip()
    data = {
        "root": flow.root.resolve().as_posix(), "graphs": {"flow": graph}, "nodes": nodes, "rel": rel, "checks": checks,
        "legend": LEGEND,
        "prune": True,
        "cards": True,
        "code": code,
        "labels": {
            "items": "값의 흐름 (출발점 → 사용 지점)",
            "status": {"bypass": "기록 안 됨", "covered": "기록됨", "outside": "사용 안 됨"},
            "cat": SOURCE,
            "kind": {"flow": "데이터 흐름"},
            "tab": {"flow": "데이터 흐름"},
            "sub": {"bypass": "로그 없이 흐른 경로", "covered": "값을 기록한 로그", "outside": "어떤 사용 지점에도 닿지 않음"},
        },
    }
    html = (Path(audit.__file__).parent / "viewer.html").read_text(encoding="utf-8")
    (out / "index.html").write_text(html.replace("__DATA__", json.dumps(data).replace("</", r"<\/")), encoding="utf-8")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("src_root", type=Path)
    p.add_argument("out_dir", type=Path)
    p.add_argument("--entry", nargs="+", required=True)
    p.add_argument("--exclude", nargs="*", default=[])
    p.add_argument("--only", nargs="*", default=[])
    a = p.parse_args()
    a.out_dir.mkdir(parents=True, exist_ok=True)
    audit_rules.PACKAGE = a.src_root.resolve().name
    funcs, classes = audit.scan(a.src_root, a.only)
    funcs = {q: i for q, i in funcs.items() if not any(x in q for x in a.exclude)}
    imports, cls_nodes = audit_rules.module_info(a.src_root, audit.sources(a.src_root, a.only))
    flow = Flow(a.src_root, funcs, classes, imports, cls_nodes, a.entry)
    rows = flow.verdicts()
    graph = draw(flow, a.out_dir, rows)
    audit.draw_cfgs({q: funcs[q] for q in {flow.info[n]["func"] for n in graph["ids"].values()} if q in funcs}, a.out_dir, [])
    write_viewer(flow, a.out_dir, graph, rows)
    with open(a.out_dir / "flow_checks.tsv", "w", encoding="utf-8") as fh:
        fh.write(TAB.join(["status", "source", "sink", "logs", "path"]) + NL)
        for r in rows:
            fh.write(TAB.join([r["status"], r["source"], r["sink"] or "", ", ".join(r["via"]), " -> ".join(r["path"])]) + NL)
    print(f"sources={len(flow.sources)} sinks={len(flow.sinks)} logs={len(flow.logs)} edges={len(flow.edges)} "
          f"checks={ {k: sum(1 for r in rows if r['status'] == k) for k in ('covered', 'bypass', 'outside')} }")


if __name__ == "__main__":
    main()

"""Dump AST, call graph and CFGs of a Python package for manual review.

The tool does not decide what is an LLM call, a tool call or a log statement.
It renders the structure as-is; the reviewer chooses what to look at via --entry / --mark / --cfg.

usage:
  python audit.py <src_root> <out_dir>
      [--entry Class.method ...]   # also draw the call graph reachable from these functions
      [--mark TEXT ...]            # highlight call-graph nodes whose name contains TEXT
      [--cfg TEXT ...]             # only draw CFGs of functions whose qualname contains TEXT (default: all)
      [--exclude TEXT ...]         # drop functions whose qualname contains TEXT from all outputs
"""

import argparse
import ast
import textwrap
from pathlib import Path

import astor
import graphviz
from py2cfg import CFGBuilder
from py2cfg.builder import CFGBuilder as _B

# ---------- py2cfg compatibility patches (rendering only, no semantics added) ----------
_orig_raise = _B.visit_Raise


def _visit_raise(self, node):
    # py2cfg crashes on bare `raise` / `raise mod.Exc(...)`: draw them as leaving the function
    exc = node.exc
    if (isinstance(exc, ast.Call) and isinstance(exc.func, ast.Name)) or isinstance(exc, ast.Name):
        return _orig_raise(self, node)
    if self.current_block.statements:
        block = self.new_block(node)
        self.current_block.add_exit(block)
        self.current_block = block
    else:
        self.current_block.add_statement(node)
    self.current_block = self.new_block(node)


_orig_call = _B.visit_Call


def _visit_call(self, node):
    # py2cfg only names simple callees (`a.b()`); label the rest (`x[0].f()`) by their source
    func = node.func
    node.func = ast.Name(id=ast.unparse(func))
    try:
        _orig_call(self, node)
    finally:
        node.func = func


_B.visit_Raise = _visit_raise
_B.visit_Call = _visit_call
astor.to_source = lambda node: ast.unparse(node) + "\n"  # astor predates `match`


# ---------- 1. AST ----------
def scan(root: Path):
    funcs, classes = {}, {}
    for f in sorted(root.rglob("*.py")):
        rel = f.relative_to(root).as_posix()
        tree = ast.parse(f.read_text(encoding="utf-8"))
        for node in tree.body:
            if isinstance(node, ast.ClassDef):
                classes[node.name] = {"bases": [ast.unparse(b).split(".")[-1] for b in node.bases], "methods": []}
                for m in node.body:
                    if isinstance(m, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        classes[node.name]["methods"].append(m.name)
                        funcs[f"{node.name}.{m.name}"] = {"node": m, "file": rel, "cls": node.name}
            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                funcs[f"{rel}#{node.name}"] = {"node": node, "file": rel, "cls": None}
    for info in funcs.values():
        calls = (c for c in ast.walk(info["node"]) if isinstance(c, ast.Call))
        info["calls"] = sorted(calls, key=lambda c: (c.lineno, c.col_offset))
    return funcs, classes


def write_ast(root: Path, funcs, classes, out: Path):
    lines, by_file = [], {}
    for q, info in funcs.items():
        by_file.setdefault(info["file"], []).append(info)
    for file in sorted(by_file):
        lines.append(file)
        for info in sorted(by_file[file], key=lambda i: i["node"].lineno):
            n, cls = info["node"], info["cls"]
            owner = f"class {cls}({', '.join(classes[cls]['bases'])}) :: " if cls else ""
            lines.append(f"  {owner}def {n.name}  L{n.lineno}-{n.end_lineno}")
            lines += [f"      L{c.lineno:<4} {ast.unparse(c.func)}()" for c in info["calls"]]
    (out / "ast_tree.txt").write_text("\n".join(lines), encoding="utf-8")
    for f in sorted(root.rglob("*.py")):
        dst = out / "ast_dump" / f.relative_to(root).with_suffix(".txt")
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_text(ast.dump(ast.parse(f.read_text(encoding="utf-8")), indent=2, include_attributes=False), encoding="utf-8")


# ---------- 2. Call graph ----------
def mro(cls: str, classes: dict) -> list[str]:
    out, todo = [], [cls]
    while todo:
        c = todo.pop(0)
        if c in classes and c not in out:
            out.append(c)
            todo += classes[c]["bases"]
    return out


def resolve(cls: str, method: str, classes: dict) -> str | None:
    return next((f"{c}.{method}" for c in mro(cls, classes) if method in classes[c]["methods"]), None)


def edges(funcs, classes):
    """(caller, callee, line, kind). kind: static | dispatch (subclass override) | super | external."""
    out = []
    for q, info in funcs.items():
        cls = info["cls"]
        for c in info["calls"]:
            f = c.func
            if cls and isinstance(f, ast.Attribute) and isinstance(f.value, ast.Name) and f.value.id == "self":
                if tgt := resolve(cls, f.attr, classes):
                    out.append((q, tgt, c.lineno, "static"))
                    out += [
                        (q, f"{sub}.{f.attr}", c.lineno, "dispatch")
                        for sub in classes
                        if sub != cls and cls in mro(sub, classes) and f"{sub}.{f.attr}" in funcs
                    ]
                    continue
            if cls and isinstance(f, ast.Attribute) and isinstance(f.value, ast.Call) and ast.unparse(f.value) == "super()":
                tgt = next((t for b in mro(cls, classes)[1:] if (t := resolve(b, f.attr, classes))), None)
                if tgt:
                    out.append((q, tgt, c.lineno, "super"))
                    continue
            if isinstance(f, ast.Name) and (tgt := next((k for k in funcs if k.endswith(f"#{f.id}")), None)):
                out.append((q, tgt, c.lineno, "static"))
                continue
            out.append((q, f"ext: {ast.unparse(f)}", c.lineno, "external"))
    return out


STYLE = {"static": {}, "super": {"color": "blue"}, "dispatch": {"style": "dashed"}, "external": {"color": "gray"}}


def draw_callgraph(funcs, es, out: Path, name: str, marks: list[str]):
    g = graphviz.Digraph(name, graph_attr={"rankdir": "LR"}, node_attr={"fontsize": "10", "shape": "box"})
    ids = {n: f"n{i}" for i, n in enumerate(sorted({n for e in es for n in e[:2]}))}  # names may hold quotes etc.
    for n, i in ids.items():
        attrs = {"style": "filled", "fillcolor": "#ffe08a"} if any(m in n for m in marks) else {}
        if n.startswith("ext: "):
            attrs = {"shape": "plaintext", "fontcolor": "gray30", **attrs}
        label = f"{n}\n{funcs[n]['file']}:{funcs[n]['node'].lineno}" if n in funcs else n
        g.node(i, label=label.replace("\\", "\\\\"), **attrs)
    for a, b, line, kind in sorted(set(es)):
        g.edge(ids[a], ids[b], label=f"L{line}", fontsize="8", **STYLE[kind])
    g.render(out / name, format="svg", cleanup=True)


def reachable(start: list[str], es) -> set[str]:
    adj = {}
    for a, b, *_ in es:
        adj.setdefault(a, set()).add(b)
    seen, todo = set(), list(start)
    while todo:
        if (n := todo.pop()) not in seen:
            seen.add(n)
            todo += adj.get(n, ())
    return seen


# ---------- 3. CFG ----------
def draw_cfgs(funcs, out: Path, filters: list[str]):
    (out / "cfg").mkdir(exist_ok=True)
    for q, info in funcs.items():
        if filters and not any(t in q for t in filters):
            continue
        cfg = CFGBuilder().build_from_src(q, textwrap.dedent(ast.unparse(info["node"])))
        fn = next(iter(cfg.functioncfgs.values()))
        fn.build_visual(str(out / "cfg" / q.replace("#", "__").replace("/", "_")), "svg", show=False, cleanup=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("src_root", type=Path)
    p.add_argument("out_dir", type=Path)
    p.add_argument("--entry", nargs="*", default=[])
    p.add_argument("--mark", nargs="*", default=[])
    p.add_argument("--cfg", nargs="*", default=[])
    p.add_argument("--exclude", nargs="*", default=[])
    a = p.parse_args()
    a.out_dir.mkdir(parents=True, exist_ok=True)
    funcs, classes = scan(a.src_root)
    funcs = {q: info for q, info in funcs.items() if not any(x in q for x in a.exclude)}
    write_ast(a.src_root, funcs, classes, a.out_dir)
    es = edges(funcs, classes)
    draw_callgraph(funcs, es, a.out_dir, "callgraph_full", a.mark)
    if a.entry:
        keep = reachable(a.entry, es)
        draw_callgraph(funcs, [e for e in es if e[0] in keep], a.out_dir, "callgraph_entry", a.mark)
    draw_cfgs(funcs, a.out_dir, a.cfg)

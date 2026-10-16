"""Statement-level CFG for one function, built for wrapper verification (used by wrapper_targets.py --cfg).

Why not py2cfg (audit.py's renderer): it does not model "a call inside `try` may raise -> except handler",
so the [LLM_ERROR] handler of a wrapper is drawn disconnected, and a function whose body is `try/except ... raise e`
comes out empty. Here every statement inside a `try` body that contains a call (or a `raise`) gets an exception
edge to each handler of the innermost `try`; uncaught `raise` goes to a RAISE exit.

Over-approximation (allowed, like the rest of the tool): `finally` is entered from every end of try/except and
leads both to the code after the try and to the exits; loop bodies may run zero or more times.

Node marks:  red = statement with an LLM call (rule hit on that line)   green = statement that logs [LLM_*]
Edge marks:  red dashed = exception   T / F = branch   loop = back edge
"""

import ast

FUNC = (ast.FunctionDef, ast.AsyncFunctionDef)
LOG_TAG = "[LLM_"


class _G:
    def __init__(self, llm_lines, log_lines):
        self.nodes, self.edges = [], []  # nodes: dict(id,label,kind,line)
        self.llm_lines, self.log_lines = llm_lines, log_lines

    def node(self, label, kind="stmt", line=None, stmt=None):
        if stmt is not None:
            lines = range(stmt.lineno, (stmt.end_lineno or stmt.lineno) + 1)
            if any(l in self.llm_lines for l in lines):
                kind = "llm"
            elif any(
                isinstance(x, ast.Constant) and isinstance(x.value, str) and LOG_TAG in x.value for x in ast.walk(stmt)
            ):
                kind = "log"
        n = {"id": f"b{len(self.nodes)}", "label": label, "kind": kind, "line": line}
        self.nodes.append(n)
        return n["id"]

    def link(self, ends, dst, label=""):
        for src, lab in ends:
            self.edges.append((src, dst, lab or label, "normal"))

    def exc(self, src, dst, label):
        self.edges.append((src, dst, label, "exception"))


def _src(stmt, header_only=False, width=70):
    if header_only:
        s = ast.unparse(stmt).split("\n", 1)[0]
    else:
        s = ast.unparse(stmt)
    s = " ".join(x.strip() for x in s.splitlines())
    return s if len(s) <= width else s[: width - 1] + "…"


def _has_call(node):
    return any(isinstance(x, (ast.Call, ast.Await, ast.Raise)) for x in ast.walk(node))


class _Builder:
    def __init__(self, g):
        self.g = g
        self.loops = []  # (header_id, break_ends list)
        self.handlers = []  # stack of [(handler_entry_id, exc_label)]
        self.finals = []  # stack of finally entry ids (or None)
        self.exit = g.node("EXIT (return)", "exit")
        self.raise_exit = g.node("RAISE (exception leaves the function)", "raise")
        self.used_raise = False

    def _propagate(self, nid, label):
        """exception from nid: to the handlers of the innermost try; if none of them catches everything,
        also on to the outer try, and finally out of the function (RAISE)."""
        for level in reversed(self.handlers):
            for h, lab, catch_all in level:
                self.g.exc(nid, h, lab)
            if any(c for _, _, c in level):
                return
        self.used_raise = True
        self.g.exc(nid, self.raise_exit, label)

    def _exc_edges(self, nid, stmt):
        if self.handlers and _has_call(stmt):  # only inside a try: elsewhere an exception simply leaves
            self._propagate(nid, "other exception")

    def _to_raise(self, nid, label="raise"):
        self._propagate(nid, label)

    def block(self, stmts, ends):
        for s in stmts:
            ends = self.stmt(s, ends)
            if not ends:
                break  # rest is unreachable
        return ends

    def stmt(self, s, ends):
        g = self.g
        L = f"L{s.lineno}  "
        if isinstance(s, ast.If):
            h = g.node(L + "if " + _src(s.test), "cond", s.lineno, s.test)
            g.link(ends, h)
            self._exc_edges(h, s.test)
            t = self.block(s.body, [(h, "T")])
            f = self.block(s.orelse, [(h, "F")]) if s.orelse else [(h, "F")]
            return t + f
        if isinstance(s, (ast.For, ast.AsyncFor, ast.While)):
            head = (
                ("while " + _src(s.test))
                if isinstance(s, ast.While)
                else (
                    ("async for " if isinstance(s, ast.AsyncFor) else "for ") + _src(s.target) + " in " + _src(s.iter)
                )
            )
            h = g.node(L + head, "cond", s.lineno, s.test if isinstance(s, ast.While) else s.iter)
            g.link(ends, h)
            self._exc_edges(h, s.test if isinstance(s, ast.While) else s.iter)
            self.loops.append((h, []))
            body_ends = self.block(s.body, [(h, "T")])
            g.link(body_ends, h, "loop")
            _, brk = self.loops.pop()
            out = self.block(s.orelse, [(h, "F")]) if s.orelse else [(h, "F")]
            return out + brk
        if isinstance(s, (ast.With, ast.AsyncWith)):
            hdr = ast.Expr(value=ast.Tuple(elts=[i.context_expr for i in s.items], ctx=ast.Load()))
            hdr.lineno, hdr.end_lineno = s.lineno, max(i.context_expr.end_lineno or s.lineno for i in s.items)
            h = g.node(L + _src(s, header_only=True), "stmt", s.lineno, hdr)
            g.link(ends, h)
            self._exc_edges(h, hdr)
            return self.block(s.body, [(h, "")])
        if isinstance(s, ast.Try) or type(s).__name__ == "TryStar":
            h = g.node(L + "try", "try", s.lineno)
            g.link(ends, h)
            fin = g.node(f"L{s.finalbody[0].lineno}  finally", "try", s.finalbody[0].lineno) if s.finalbody else None
            hs = []
            for hd in s.handlers:
                typ = _src(hd.type) if hd.type is not None else "any exception"
                hid = g.node(f"L{hd.lineno}  except {typ}" + (f" as {hd.name}" if hd.name else ""), "except", hd.lineno)
                hs.append((hid, typ, hd))
            if not hs and fin:  # try/finally: an exception goes through finally (then re-raised)
                hs_edges = [(fin, "exception", True)]
            else:
                hs_edges = [
                    (hid, typ, hd.type is None or _src(hd.type) in ("Exception", "BaseException"))
                    for hid, typ, hd in hs
                ]
            self.handlers.append(hs_edges)
            self.finals.append(fin)
            body_ends = self.block(s.body, [(h, "")])
            self.handlers.pop()
            body_ends = self.block(s.orelse, body_ends) if s.orelse else body_ends
            out = list(body_ends)
            for hid, _, hd in hs:
                out += self.block(hd.body, [(hid, "")])
            self.finals.pop()
            if fin:
                g.link(out, fin)
                fin_ends = self.block(s.finalbody, [(fin, "")])
                for src, _ in fin_ends:  # finally may also be on the way out of a return / raise
                    g.link([(src, "")], self.exit, "then return")
                    self._to_raise(src, "then re-raise")
                return fin_ends
            return out
        if isinstance(s, ast.Match):
            h = g.node(L + "match " + _src(s.subject), "cond", s.lineno, s.subject)
            g.link(ends, h)
            out = []
            for c in s.cases:
                out += self.block(c.body, [(h, "case " + _src(c.pattern, width=30))])
            return out + [(h, "no match")]
        if isinstance(s, ast.Return):
            n = g.node(L + _src(s), "stmt", s.lineno, s)
            g.link(ends, n)
            self._exc_edges(n, s)
            fin = next((f for f in reversed(self.finals) if f), None)
            g.link([(n, "")], fin or self.exit, "return")
            return []
        if isinstance(s, ast.Raise):
            n = g.node(L + _src(s), "stmt", s.lineno, s)
            g.link(ends, n)
            self._to_raise(n)
            return []
        if isinstance(s, ast.Break):
            n = g.node(L + "break", "stmt", s.lineno)
            g.link(ends, n)
            if self.loops:
                self.loops[-1][1].append((n, "break"))
            return []
        if isinstance(s, ast.Continue):
            n = g.node(L + "continue", "stmt", s.lineno)
            g.link(ends, n)
            if self.loops:
                g.link([(n, "")], self.loops[-1][0], "continue")
            return []
        if isinstance(s, (*FUNC, ast.ClassDef)):
            n = g.node(L + ("def " if isinstance(s, FUNC) else "class ") + s.name + " (nested)", "stmt", s.lineno)
            g.link(ends, n)
            return [(n, "")]
        # simple statement
        n = g.node(L + _src(s), "stmt", s.lineno, s)
        g.link(ends, n)
        self._exc_edges(n, s)
        return [(n, "")]


def build(func: ast.AST, llm_lines=(), title=""):
    """-> (nodes, edges) for one function. llm_lines: source lines of the LLM calls in this function."""
    g = _G(set(llm_lines), set())
    b = _Builder(g)
    entry = g.node(f"L{func.lineno}  {title or func.name}()", "entry", func.lineno)
    body = func.body
    if (
        body
        and isinstance(body[0], ast.Expr)
        and isinstance(body[0].value, ast.Constant)
        and isinstance(body[0].value.value, str)
    ):
        body = body[1:]  # docstring
    ends = b.block(body, [(entry, "")])
    g.link(ends, b.exit, "end")
    if not b.used_raise and not any(e[1] == b.raise_exit for e in g.edges):
        g.nodes = [n for n in g.nodes if n["id"] != b.raise_exit]
    return g.nodes, g.edges


STYLE = {
    "entry": 'shape=box, style="rounded,filled", fillcolor="#dbe6ff", color="#3355aa"',
    "exit": 'shape=box, style="rounded,filled", fillcolor="#eeeeee", color="#888888"',
    "raise": 'shape=box, style="rounded,filled", fillcolor="#f6e0e0", color="#a05050"',
    "stmt": 'shape=box, color="#888888"',
    "cond": 'shape=box, style="rounded", color="#666666", penwidth=1.4',
    "try": 'shape=box, style="rounded,dashed", color="#666666"',
    "except": 'shape=box, style="rounded,filled", fillcolor="#fff3d6", color="#c08a00"',
    "llm": 'shape=box, style="filled,bold", fillcolor="#ffe0e0", color="#d00000", penwidth=2.5, fontcolor="#a00000"',
    "log": 'shape=box, style="filled", fillcolor="#e3f5e7", color="#1a9e3a", penwidth=2',
}


def to_dot(nodes, edges, name="cfg"):
    esc = lambda s: s.replace("\\", "\\\\").replace('"', '\\"')  # noqa: E731
    out = [
        f'digraph "{esc(name)}" {{',
        "rankdir=TB; nodesep=0.3; ranksep=0.35;",
        'node [fontname="Menlo, Consolas, monospace", fontsize=10, margin="0.12,0.05"]; edge [fontsize=9, color="#777777"];',
    ]
    for n in nodes:
        out.append(f'{n["id"]} [label="{esc(n["label"])}", {STYLE[n["kind"]]}];')
    for a, b, lab, kind in edges:
        attrs = [f'label="{esc(lab)}"'] if lab else []
        if kind == "exception":
            attrs += ["style=dashed", 'color="#d00000"', 'fontcolor="#d00000"']
        elif lab == "loop" or lab == "continue":
            attrs += ["constraint=false", 'color="#999999"']
        out.append(f"{a} -> {b} [{', '.join(attrs)}];")
    out.append("}")
    return "\n".join(out)

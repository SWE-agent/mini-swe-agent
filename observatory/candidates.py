# /// script
# requires-python = ">=3.10"
# dependencies = []
# ///
"""The calls an anchor can be, for the anchors session: every call in the chosen files whose implementation these files
do not show. Read with the checker's own name resolution (flow_verify.Verifier), never decided by the called name.

  candidate      an imported library or a module outside the chosen files (litellm.completion, subprocess.run), a
                 builtin with an effect (input, exec), a method of an object stored on self (self.model.query) or
                 held at module level (prompt_session.prompt), a function looked up by key, passed in or held in a
                 variable (tools[name](...), callback(...))
  not one        a function or class of the chosen files (followed inside instead), the logger, a builtin or library
                 call that only computes a value (len, json.loads), a change of a list or dict (messages.append), or a
                 method of a value computed in the function (reply.get(...))

usage (repo root):  uv run observatory/candidates.py src/pkg/agent.py src/pkg/tools.py
"""

import argparse
import ast
import builtins
from pathlib import Path

from flow_verify import Verifier, pure_library


class Candidates(Verifier):
    def __init__(self, root: Path, files: list[str]):
        self.root, self.anchors, self.logger = root, {"sources": []}, "observatory_log"
        self.trees = {f: ast.parse((root / f).read_text(encoding="utf-8")) for f in files}
        self.modules = {self.module_id(f) for f in self.trees}
        self.funcs, self.classes, self.module_funcs, self.imports = {}, {}, {}, {}
        self.parent, self.own_names, self.reaching = {}, {}, set()
        self.var_types, self.field_types, self.picked = {}, {}, set()
        for file, tree in self.trees.items():
            self.scan(file, tree)
        for q, f in self.funcs.items():
            self.enter(q)
            for s in ast.walk(f["node"]):
                if isinstance(s, ast.Assign):
                    for t in s.targets:
                        if self.is_self_attr(t) and (typ := self.type_of(s.value)):
                            self.field_types[(f["cls"], t.attr)] = typ
                        elif isinstance(t, ast.Name) and isinstance(s.value, ast.Subscript):
                            self.picked.add((q, t.id))
                        elif isinstance(t, ast.Name) and (typ := self.type_of(s.value)):
                            self.var_types[(q, t.id)] = typ

    def params(self, q: str) -> set[str]:
        a = self.funcs[q]["node"].args
        return {p.arg for p in [*a.posonlyargs, *a.args, *a.kwonlyargs, a.vararg, a.kwarg] if p} - {"self", "cls"}

    def held(self, e) -> bool:
        """An object the function did not compute: stored on self (not plain data), at module level, or a parameter."""
        while isinstance(e, (ast.Attribute, ast.Subscript)) and not self.is_self_attr(e):
            e = e.value
        if self.is_self_attr(e):
            return not self.data_field(e.attr)
        if isinstance(e, ast.Name):
            stored = {
                n.id
                for n in ast.walk(self.funcs[self.q]["node"])
                if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Store)
            }
            return e.id in self.params(self.q) or e.id not in stored
        return False

    def kind(self, c: ast.Call) -> str | None:
        kind = self.callee(c.func)
        if kind[0] == "ext":
            return None if pure_library(kind[1]) else "builtin" if hasattr(builtins, kind[1]) else "imported"
        if kind[0] != "value":
            return None
        if isinstance(c.func, ast.Name):
            return "callable value"
        if isinstance(c.func, ast.Subscript) or self.held(c.func.value):
            return "object method"
        return None

    def rows(self) -> list[dict]:
        out = []
        for q, f in self.funcs.items():
            if q in self.parent:
                continue  # read with the function it is defined in
            self.enter(q)
            lines = (self.root / f["file"]).read_text(encoding="utf-8").splitlines()
            for c in ast.walk(f["node"]):
                if isinstance(c, ast.Call) and (k := self.kind(c)):
                    out.append(
                        {
                            "path": f["file"],
                            "line": c.lineno,
                            "in": q,
                            "call": ast.unparse(c.func),
                            "kind": k,
                            "code": lines[c.lineno - 1].strip(),
                        }
                    )
        return sorted(
            {(r["path"], r["line"], r["call"]): r for r in out}.values(), key=lambda r: (r["path"], r["line"])
        )


def listing(root: Path, files: list[str]) -> str:
    return "\n".join(
        f"{i}. {r['path']}:{r['line']}  `{r['call']}`  in {r['in']} ({r['kind']})\n     {r['code']}"
        for i, r in enumerate(Candidates(root, files).rows(), start=1)
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("files", nargs="+")
    print(listing(Path.cwd(), p.parse_args().files))

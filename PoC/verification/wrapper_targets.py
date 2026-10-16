"""Wrapper targets: find every LLM call with the rules R1-R3 (llm_rules.py) and draw ONLY what needs a wrapper.

  red double circle   the LLM call itself (rule, pattern, line)
  red box             the function that makes that call  -> wrapper target
  green border        a function that already logs an [LLM_*] boundary (an existing wrapper)
  grey box            (--entry only) functions on the shortest path from the entry to a target, for context

Pure AST, no LLM. False positives are allowed, false negatives are not:
  - calls AND function references (`retryer(client.chat.completions.create, ...)`) are checked,
  - module-level code and class bodies are scanned too,
  - `self.model.query()`-style calls are linked to every function of that name (dynamic dispatch),
  - third-party imports that no rule covers are listed in unreviewed_imports.tsv (safety net for new SDKs).

usage:
  python PoC/verification/wrapper_targets.py src/minisweagent PoC/verification/out_wrapper --entry DefaultAgent.run
  python wrapper_targets.py <smolagents>/src/smolagents out --entry MultiStepAgent.run
  python wrapper_targets.py <browser-use>/browser_use out --entry agent/service.py::Agent.run
  python wrapper_targets.py <browser-use>/browser_use out --through agent/service.py::Agent.get_model_output

  --entry    start of the coverage check (and grey context paths)
  --through  scope: keep only LLM calls reachable FROM these functions, and check coverage from them
  --rules    subset of R1,R2,R3 (default: all)

outputs (out_dir):
  wrapper_targets.svg / .png / .dot   the picture
  wrapper_targets.tsv                 one row per LLM call: function, rule, whether an [LLM_*] log covers it on every
                                      path from the entry, and one path that bypasses the log (if any)
  report.html                         open in a browser: click a target in the list or the graph -> its CFG + verdict
  cfg/<function>.svg, cfg_targets.tsv with --cfg: CFGs (cfg_builder.py, with exception edges) of the wrap targets and
                                      of the existing [LLM_*] wrappers on their paths (needs only Graphviz)
  unreviewed_imports.tsv              third-party libraries no rule covers: review, then add to llm_rules.py
                                      (LIB_RULES if it calls an LLM, REVIEWED_NON_LLM if not)
"""

import argparse
import ast
import shutil
import subprocess
import sys
from collections import deque
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import llm_rules  # noqa: E402

FUNC = (ast.FunctionDef, ast.AsyncFunctionDef)
LOG_TAG = "[LLM_"


# ---------------------------------------------------------------- scan (file-qualified keys)
def scan(root: Path):
    trees = {f.relative_to(root).as_posix(): ast.parse(f.read_text(encoding="utf-8")) for f in sorted(root.rglob("*.py"))}
    classes = {}
    for rel, tree in trees.items():
        for c in (n for n in tree.body if isinstance(n, ast.ClassDef)):
            classes.setdefault(c.name, []).append({"file": rel, "bases": [ast.unparse(b).split(".")[-1] for b in c.bases],
                                                   "methods": [m.name for m in c.body if isinstance(m, FUNC)],
                                                   "strings": llm_rules.self_string_attrs(c)})
    funcs, sites = {}, []
    for rel, tree in trees.items():
        imports = llm_rules.imported_roots(tree)
        units = [(None, n.name, n.lineno, [n]) for n in tree.body if isinstance(n, FUNC)]
        module = []
        for n in tree.body:
            if isinstance(n, ast.ClassDef):
                units += [(n.name, m.name, m.lineno, [m]) for m in n.body if isinstance(m, FUNC)]
                module += [b for b in n.body if not isinstance(b, FUNC)]
            elif not isinstance(n, FUNC):
                module.append(n)
        units.append((None, "<module>", 1, module))
        for cls, name, line, body in units:
            q = f"{rel}::{cls}.{name}" if cls else f"{rel}::{name}"
            calls = [c for b in body for c in ast.walk(b) if isinstance(c, ast.Call)]
            logs = any(isinstance(x, ast.Constant) and isinstance(x.value, str) and LOG_TAG in x.value
                       for c in calls for x in ast.walk(c))
            funcs[q] = {"file": rel, "line": line, "cls": cls, "name": name, "calls": calls, "imports": imports, "logs": logs,
                        "node": body[0] if name != "<module>" else None}
            strings = {}
            for k in reversed(mro(cls, classes)) if cls else []:
                for rec in classes[k]:
                    strings.update(rec["strings"])
            for c in calls:
                if hit := llm_rules.classify(c, imports, strings):
                    sites.append({"func": q, "line": c.lineno, "rule": hit[0], "pattern": hit[1], "detail": hit[2]})
            for c in (r for b in body for r in llm_rules.references(b)):
                if (hit := llm_rules.classify(c, imports, strings)) and hit[0] != "R3-http":
                    sites.append({"func": q, "line": c.lineno, "rule": hit[0] + "-ref", "pattern": hit[1], "detail": hit[2]})
    return funcs, classes, sorted(sites, key=lambda s: (funcs[s["func"]]["file"], s["line"]))


def mro(cls, classes):
    out, todo = [], [cls]
    while todo:
        c = todo.pop(0)
        if c in classes and c not in out:
            out.append(c)
            for rec in classes[c]:
                todo += rec["bases"]
    return out


def method_keys(cls, method, classes, prefer_file=None):
    for k in mro(cls, classes):
        recs = [r for r in classes[k] if method in r["methods"]]
        if recs:
            own = [r for r in recs if r["file"] == prefer_file] if k == cls else []
            return [f"{r['file']}::{k}.{method}" for r in (own or recs)]
    return []


def edges(funcs, classes, local):
    """(caller, callee, weight). weight 0 = statically resolved, 1 = guessed by name (dynamic dispatch)."""
    by_name = {}
    for q, i in funcs.items():
        by_name.setdefault(i["name"], []).append(q)
    out = set()
    for q, info in funcs.items():
        cls = info["cls"]
        for c in info["calls"]:
            ch = llm_rules.chain_of(c.func)
            if not ch:
                continue
            if cls and ch[0] == "self" and len(ch) == 2:
                tgts = method_keys(cls, ch[1], classes, info["file"])
                if tgts:
                    out |= {(q, t, 0) for t in tgts}
                    out |= {(q, t, 1) for sub in classes if sub != cls and cls in mro(sub, classes)
                            for t in method_keys(sub, ch[1], classes) if t.split("::")[1].startswith(sub + ".")}
                    continue
            if cls and ch[0] == "super()" and len(ch) == 2:  # super().m(): base classes only
                bases = [b for rec in classes.get(cls, []) if rec["file"] == info["file"] for b in rec["bases"]]
                tgts = [t for b in bases for t in method_keys(b, ch[1], classes)]
                out |= {(q, t, 0) for t in tgts}
                continue  # no local base defines it -> the base is third-party (Exception, BaseModel ...)
            if len(ch) == 1 and ch[0] in classes:
                out |= {(q, t, 0) for t in method_keys(ch[0], "__init__", classes)}
                continue
            root_pkg = info["imports"].get(ch[0])
            if root_pkg and root_pkg not in local:
                continue
            out |= {(q, t, 1) for t in by_name.get(ch[-1], []) if t != q}  # weight 1 = guessed by name
    return out


# ---------------------------------------------------------------- analysis
def shortest_paths(entry, es, blocked=frozenset()):
    adj = {}
    for a, b, w in es:
        adj.setdefault(a, []).append((b, w))
    prev, dist, dq = {entry: None}, {entry: 0}, deque([entry])
    while dq:
        n = dq.popleft()
        if n in blocked and n != entry:
            continue
        for b, w in sorted(adj.get(n, [])):
            if dist[n] + w < dist.get(b, 1 << 30):
                dist[b], prev[b] = dist[n] + w, n
                (dq.appendleft if w == 0 else dq.append)(b)
    return prev


def find(spec, funcs, classes):
    m = [k for k in funcs if k == spec or k.endswith("::" + spec)]
    if not m and "." in spec:
        m = method_keys(*spec.rsplit(".", 1), classes)[:1]
    return m


def render(dot_path: Path):
    if shutil.which("dot"):
        for fmt in ("svg", "png"):
            subprocess.run(["dot", f"-T{fmt}", str(dot_path), "-o", str(dot_path.with_suffix("." + fmt))], check=True)
        return True
    try:
        import graphviz
        for fmt in ("svg", "png"):
            graphviz.Source(dot_path.read_text(encoding="utf-8")).render(dot_path.with_suffix(""), format=fmt, cleanup=True)
        return True
    except Exception:
        return False


def main():
    p = argparse.ArgumentParser()
    p.add_argument("src_root", type=Path)
    p.add_argument("out_dir", type=Path)
    p.add_argument("--entry", nargs="*", default=[], help="e.g. DefaultAgent.run (adds grey context paths)")
    p.add_argument("--rules", default="R1,R2,R3")
    p.add_argument("--through", nargs="*", default=[],
                   help="scope: keep only LLM calls reachable FROM these functions (e.g. Agent.get_model_output), "
                        "and judge coverage only on paths that pass through them")
    p.add_argument("--cfg", action="store_true",
                   help="also draw CFGs (audit.py's renderer, needs py2cfg) of the wrap targets and of the "
                        "[LLM_*]-logging functions on their paths -> out_dir/cfg, listed in cfg_targets.tsv")
    a = p.parse_args()
    llm_rules.RULES = set(a.rules.split(","))
    a.out_dir.mkdir(parents=True, exist_ok=True)
    local = {a.src_root.resolve().name} | {x.stem for x in a.src_root.rglob("*.py")} | {x.name for x in a.src_root.rglob("*") if x.is_dir()}

    funcs, classes, sites = scan(a.src_root)
    es = edges(funcs, classes, local)

    if a.through:  # scope restriction: only LLM calls that these functions can reach
        thr = [k for spec in a.through for k in find(spec, funcs, classes)]
        if not thr:
            sys.exit(f"--through not found: {a.through}")
        in_scope = set()
        for t in thr:
            in_scope |= set(shortest_paths(t, es))
        sites = [s for s in sites if s["func"] in in_scope]
        print(f"scope (--through {', '.join(x.split('::')[1] for x in thr)}): {len(sites)} LLM calls kept")

    targets = sorted({s["func"] for s in sites})
    direct_log = {q for q, i in funcs.items() if i["logs"]}
    # helper pattern (smolagents: `self._log_llm_query(...)`): the caller of a pure logging helper is the wrapper.
    # Only statically resolved edges (weight 0) and only helpers that do not reach an LLM call themselves.
    reaches_llm = set(targets)
    rev0 = {}
    for x, y, w in es:
        rev0.setdefault(y, set()).add(x)
    todo = list(reaches_llm)
    while todo:
        for c in rev0.get(todo.pop(), ()):
            if c not in reaches_llm:
                reaches_llm.add(c)
                todo.append(c)
    helpers = {q for q in direct_log if q not in reaches_llm}
    via_helper = {x for x, y, w in es if w == 0 and y in helpers}
    logged = (direct_log - helpers) | via_helper

    entries = []
    for spec in (a.entry + a.through if a.through else a.entry):
        m = find(spec, funcs, classes)
        if not m:
            sys.exit(f"entry not found: {spec}")
        entries += m

    # context: shortest entry -> target paths; coverage: is every path blocked by an [LLM_*]-logging function?
    context, covered, bypass = set(), {}, {}
    for e in entries:
        prev = shortest_paths(e, es)
        for t in targets:
            n = t
            while n in prev and prev[n]:
                context.add((prev[n], n))
                n = prev[n]
        reach_without_log = shortest_paths(e, es, blocked=frozenset(logged - {e}))
        for t in targets:
            if t in prev:
                ok = e in logged or t in logged or t not in reach_without_log  # the entry itself may be the wrapper
                covered[t] = covered.get(t, True) and ok
                if not ok and t not in bypass:
                    path, n = [], t
                    while n in reach_without_log and reach_without_log[n]:
                        path.append(n.split("::")[1])
                        n = reach_without_log[n]
                    bypass[t] = " <- ".join(path + [n.split("::")[1]])

    with open(a.out_dir / "wrapper_targets.tsv", "w", encoding="utf-8") as fh:
        fh.write("location\tfunction\trule\tpattern\tdetail\thas_own_LLM_log\tLLM_log_on_every_path_from_entry\tpath_without_log\n")
        for s in sites:
            f = funcs[s["func"]]
            cov = {True: "yes", False: "NO"}.get(covered.get(s["func"]), "-")
            fh.write(f"{f['file']}:{s['line']}\t{s['func']}\t{s['rule']}\t{s['pattern']}\t{s['detail']}\t"
                     f"{'yes' if s['func'] in logged else 'no'}\t{cov}\t{bypass.get(s['func'], '')}\n")

    ids = {}
    nid = lambda n: ids.setdefault(n, f"n{len(ids)}")  # noqa: E731
    g = ["digraph W {", 'rankdir=LR; nodesep=0.25; node [fontname=Helvetica, fontsize=10, shape=box]; edge [color="#999999"];']
    nodes = set(targets) | {x for e in context for x in e}
    for n in sorted(nodes):
        f = funcs[n]
        label = f"{n.split('::')[1]}\\n{f['file']}:{f['line']}"
        if n in targets:
            style = 'style="filled,bold", fillcolor="#ffe0e0", color="#d00000", penwidth=3, fontcolor="#a00000"'
            label = "WRAP TARGET\\n" + label
        elif n in entries:
            style = 'style=filled, fillcolor="#dbe6ff", color="#3355aa"'
        else:
            style = 'color="#bbbbbb", fontcolor="#555555"'
        if n in logged:
            style += ', peripheries=2, color="#1a9e3a"'
            label += "\\n(has [LLM_*] log)"
        g.append(f'{nid(n)} [label="{label}", {style}];')
    for a_, b_ in sorted(context):
        g.append(f"{nid(a_)} -> {nid(b_)};")
    for k, s in enumerate(sites):
        detail = s["detail"].replace('"', "'").replace("\\", "\\\\")[:70]
        g.append(f's{k} [shape=ellipse, peripheries=2, color="#d00000", penwidth=2.5, fontcolor="#a00000", fontsize=9, '
                 f'label="{s["rule"]}  L{s["line"]}\\n{s["pattern"]}\\n{detail}"];')
        g.append(f'{nid(s["func"])} -> s{k} [color="#d00000", penwidth=2];')
    g.append('subgraph cluster_legend { label="legend"; fontsize=10; color="#cccccc";')
    g.append('l1 [label="LLM call (R1/R2/R3)", shape=ellipse, peripheries=2, color="#d00000", fontcolor="#a00000", fontsize=9];')
    g.append('l2 [label="WRAP TARGET\\n(function making the call)", style="filled,bold", fillcolor="#ffe0e0", color="#d00000", penwidth=3, fontsize=9];')
    g.append('l3 [label="already logs [LLM_*]", peripheries=2, color="#1a9e3a", fontsize=9];')
    g.append('l4 [label="path from entry", color="#bbbbbb", fontsize=9]; l1 -> l2 -> l3 -> l4 [style=invis]; }')
    g.append("}")
    dot = a.out_dir / "wrapper_targets.dot"
    dot.write_text("\n".join(g), encoding="utf-8")
    drawn = render(dot)

    # safety net: third-party imports that no rule covers -> a person decides (no silent miss for a new SDK)
    covered_libs = llm_rules.RULE_LIBS | set(llm_rules.LIB_RULES) | set(llm_rules.HTTP_LIBS) | llm_rules.REVIEWED_NON_LLM
    unreviewed = {}
    for q, i in funcs.items():
        for name, root_pkg in i["imports"].items():
            if root_pkg and root_pkg not in local and root_pkg not in sys.stdlib_module_names and root_pkg not in covered_libs:
                unreviewed.setdefault(root_pkg, set()).add(i["file"])
    with open(a.out_dir / "unreviewed_imports.tsv", "w", encoding="utf-8") as fh:
        fh.write("library\tfiles\n")
        for lib in sorted(unreviewed):
            fh.write(lib + "\t" + ", ".join(sorted(unreviewed[lib])) + "\n")

    cfg_files, roles = {}, {}
    if a.cfg:  # CFGs of what has to be checked: the wrap targets + the existing wrappers on their paths
        on_paths = {x for e in context for x in e}
        roles = {t: "wrap target (contains an LLM call)" for t in targets}
        for q in sorted(logged):
            if not entries or q in on_paths:
                roles[q] = roles.get(q, "") + (" + " if q in roles else "") + "existing wrapper ([LLM_*] log)"
        import cfg_builder  # noqa: E402  (same folder; draws exception edges, which py2cfg does not)
        drawable = {q: funcs[q] for q in roles if funcs[q]["node"] is not None}
        safe = {q: q.replace("::", "__").replace("/", "_").replace("<", "").replace(">", "") for q in drawable}
        failed = {}
        (a.out_dir / "cfg").mkdir(exist_ok=True)
        for q in drawable:  # one by one: a failure on one function must not stop the others
            try:
                llm_lines = [s["line"] for s in sites if s["func"] == q]
                dot_src = cfg_builder.to_dot(*cfg_builder.build(funcs[q]["node"], llm_lines, q.split("::")[1]), q)
                p_dot = a.out_dir / "cfg" / f"{safe[q]}.dot"
                p_dot.write_text(dot_src, encoding="utf-8")
                if shutil.which("dot"):
                    subprocess.run(["dot", "-Tsvg", str(p_dot), "-o", str(p_dot.with_suffix(".svg"))], check=True)
                else:
                    import graphviz
                    graphviz.Source(dot_src).render(p_dot.with_suffix(""), format="svg", cleanup=True)
            except Exception as ex:  # noqa: BLE001
                failed[q] = f"CFG failed: {type(ex).__name__}"
        with open(a.out_dir / "cfg_targets.tsv", "w", encoding="utf-8") as fh:
            fh.write("function\tlocation\trole\tcfg_file\n")
            for q in sorted(roles, key=lambda q: (not roles[q].startswith("wrap"), q)):
                f = funcs[q]
                cfg_file = (failed[q] if q in failed else f"cfg/{safe[q]}.svg") if q in safe else "(module-level code: no CFG)"
                cfg_files[q] = f"cfg/{safe[q]}.svg" if q in safe and q not in failed else (failed.get(q) or False)
                fh.write(f"{q}\t{f['file']}:{f['line']}\t{roles[q]}\t{cfg_file}\n")
        if failed:
            print(f"CFG could not be drawn for {len(failed)} function(s): " + ", ".join(failed))
        print(f"CFGs drawn: {len(drawable) - len(failed)} (wrap targets {sum(1 for r in roles.values() if r.startswith('wrap'))}, "
              f"existing wrappers {sum(1 for r in roles.values() if 'existing' in r)}) -> cfg/, cfg_targets.tsv")

    # interactive report: click a node -> CFG + verdict
    import report  # noqa: E402  (same folder)
    rep_nodes = {}
    for q in sorted(set(nodes) | set(roles)):
        f = funcs[q]
        rep_nodes[q] = {"label": q.split("::")[1], "file": f["file"], "line": f["line"],
                        "kind": "target" if q in targets else ("wrapper" if q in logged else "path"),
                        "logged": q in logged, "covered": covered.get(q) if q in targets else None,
                        "bypass": bypass.get(q, ""), "cfg": cfg_files.get(q),
                        "sites": [{"line": s["line"], "rule": s["rule"], "pattern": s["pattern"], "detail": s["detail"]}
                                  for s in sites if s["func"] == q]}
    nid_to_key = {v: k for k, v in ids.items()} | {f"s{k}": s["func"] for k, s in enumerate(sites)}
    report.write(a.out_dir, f"Wrapper targets · {a.src_root.resolve().name}",
                 {"sites": len(sites), "targets": len(targets), "entry": ", ".join(x.split("::")[1] for x in entries),
                  "covered": sum(1 for t in targets if covered.get(t) is True),
                  "not_covered": sum(1 for t in targets if covered.get(t) is False)},
                 rep_nodes, nid_to_key)

    print(f"LLM calls: {len(sites)}   wrapper targets (functions): {len(targets)}")
    print(f"third-party libraries not covered by any rule (review unreviewed_imports.tsv): {len(unreviewed)}")
    if entries:
        miss = [t for t in targets if covered.get(t) is False]
        print(f"targets reachable from entry: {len(covered)}   covered by an [LLM_*] log on every path: "
              f"{sum(covered.values())}   NOT covered: {len(miss)}")
    if not drawn:
        print("Graphviz not found: open wrapper_targets.dot with any Graphviz viewer")


if __name__ == "__main__":
    main()

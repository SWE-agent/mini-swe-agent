"""audit.py + LLM query rules (R1-R3): mark, on the AST call graph, what has to be checked for a wrapper.

It reuses audit.py's scan() / edges() / draw_cfgs() unchanged, applies llm_rules.classify() to every call
(and to every function reference, e.g. `retryer(client.chat.completions.create)`), and redraws the call graph
with the rule hits highlighted:

  red double circle    the LLM call itself (R1 SDK / R2 library / R3 HTTP), with rule, pattern and line
  red filled box       a function that contains an LLM call           -> CFG target (wrapper expected here)
  red dashed box       a function that can reach one of those         -> CFG target (wrapper may be here)
  light pink dashed    reaches one only via 2+ same-name links        -> weaker evidence, check the path
  blue filled box      an --entry function
  blue dotted edge     name-based link added for calls audit.py leaves external (e.g. self.model.query ->
                       every `query` in the package), so dynamic dispatch does not hide a path

usage:
  python audit_rules.py <src_root> <out_dir> [--entry Class.method ...] [--rules R1,R2,R3]
                        [--cfg] [--view-max-links N] [--exclude TEXT ...] [--only PATH ...]

  python PoC/verification/audit_rules.py src/minisweagent PoC/verification/out_rules \
      --entry DefaultAgent.run --view-max-links 1 --cfg

  Scan the whole package (e.g. src/minisweagent), not only agents/: the LLM calls live in models/.
  Needs the same environment as audit.py (py2cfg, graphviz, astor).

outputs (out_dir):
  rule_sites.tsv               every LLM call found: location, function, rule, pattern, detail
  cfg_targets.tsv              every function to check with a CFG, why, and how many name-links it needs
  callgraph_rules_targets.svg  only the CFG targets and their LLM calls (start here)
  callgraph_rules_entry.svg    everything reachable from --entry, with the marks
  callgraph_rules_full.svg     the whole call graph, with the marks
  cfg/<function>.svg           with --cfg: CFGs of the CFG targets only (audit.py's renderer)
"""

import argparse
import ast
import sys
from collections import deque
from pathlib import Path

import graphviz

sys.path.insert(0, str(Path(__file__).resolve().parent))
import audit  # noqa: E402  (same folder, unchanged)
import llm_rules  # noqa: E402

EDGE_STYLE = {**audit.STYLE, "name": {"style": "dotted", "color": "#3355aa"}}
FUNC = (ast.FunctionDef, ast.AsyncFunctionDef)


def module_info(root: Path, files):
    """Per file: imported roots and, per class, its `self.x = "..."` strings (for R3 URL resolution)."""
    imports, cls_nodes = {}, {}
    for f in files:
        rel = f.relative_to(root).as_posix()
        tree = ast.parse(f.read_text(encoding="utf-8"))
        imports[rel] = llm_rules.imported_roots(tree)
        for n in tree.body:
            if isinstance(n, ast.ClassDef):
                cls_nodes[n.name] = n
    return imports, cls_nodes


def class_strings(cls, classes, cls_nodes):
    out = {}
    for k in reversed(audit.mro(cls, classes)) if cls else []:
        out.update(llm_rules.self_string_attrs(cls_nodes.get(k)))
    return out


def _hits(body_nodes, calls, imports, strings):
    """Rule hits for the calls, plus function references (rule suffixed with -ref, R1/R2 only)."""
    for c in calls:
        if hit := llm_rules.classify(c, imports, strings):
            yield c, hit[0], hit
    for c in (r for b in body_nodes for r in llm_rules.references(b)):
        if (hit := llm_rules.classify(c, imports, strings)) and hit[0] != "R3-http":
            yield c, hit[0] + "-ref", hit


def find_sites(root, files, funcs, classes, imports, cls_nodes):
    sites = []
    for q, info in funcs.items():
        strings = class_strings(info["cls"], classes, cls_nodes)
        body = [info["node"]] if info.get("node") is not None else []
        for c, rule, hit in _hits(body, info["calls"], imports[info["file"]], strings):
            sites.append({"func": q, "file": info["file"], "line": c.lineno, "rule": rule,
                          "pattern": hit[1], "detail": hit[2], "ext": f"ext: {ast.unparse(c.func)}"})
    # audit.py only scans def bodies: calls at module level / in class bodies would be invisible -> report them
    for f in files:
        rel = f.relative_to(root).as_posix()
        tree = ast.parse(f.read_text(encoding="utf-8"))
        body = [n for n in tree.body if not isinstance(n, (*FUNC, ast.ClassDef))]
        body += [b for n in tree.body if isinstance(n, ast.ClassDef) for b in n.body if not isinstance(b, FUNC)]
        calls = [c for b in body for c in ast.walk(b) if isinstance(c, ast.Call)]
        for c, rule, hit in _hits(body, calls, imports[rel], {}):
            q = f"{rel}#<module>"
            funcs.setdefault(q, {"file": rel, "cls": None, "calls": [], "node": None, "line": 1})
            sites.append({"func": q, "file": rel, "line": c.lineno, "rule": rule, "pattern": hit[1],
                          "detail": hit[2], "ext": f"ext: {ast.unparse(c.func)}"})
    uniq = {(s["func"], s["line"], s["rule"].removesuffix("-ref"), s["detail"]): s for s in sites}
    return sorted(uniq.values(), key=lambda s: (s["file"], s["line"]))


def bridge_edges(es, funcs, imports):
    """audit.py leaves `self.model.query()` as an external node. Link it to every package function of that
    name (over-approximation), unless the call starts at an imported third-party module (subprocess.run ...)."""
    by_name = {}
    for q in funcs:
        by_name.setdefault(q.split("#")[-1].split(".")[-1], []).append(q)
    out = set()
    for caller, callee, line, kind in es:
        if kind != "external":
            continue
        chain = callee[len("ext: "):].split("(")[0].split(".")
        root_pkg = imports.get(funcs[caller]["file"], {}).get(chain[0])
        if root_pkg and root_pkg != PACKAGE:
            continue
        for tgt in by_name.get(chain[-1], []):
            if tgt != caller:
                out.add((caller, tgt, line, "name"))
    return out


def reverse_closure(start, es):
    """Every caller of `start`, with the fewest name-based links needed to get there (0-1 BFS backwards).
    0 = reached through statically resolved calls only; 1 = typical dynamic dispatch (self.model.query);
    2+ = needs several same-name links: weaker evidence, check the path (still kept: no misses)."""
    rev = {}
    for a, b, _, kind in es:
        rev.setdefault(b, []).append((a, 1 if kind == "name" else 0))
    dist, dq = {n: 0 for n in start}, deque(start)
    while dq:
        n = dq.popleft()
        for c, w in rev.get(n, ()):
            if dist[n] + w < dist.get(c, 1 << 30):
                dist[c] = dist[n] + w
                (dq.appendleft if w == 0 else dq.append)(c)
    return dist


def draw(name, out: Path, funcs, es, sites, site_funcs, targets, entries, keep=None):
    g = graphviz.Digraph(name, graph_attr={"rankdir": "LR", "fontname": "Helvetica"},
                         node_attr={"fontsize": "10", "shape": "box", "fontname": "Helvetica"})
    es = [e for e in es if keep is None or (e[0] in keep and e[1] in keep)]
    site_ext = {s["ext"] for s in sites}
    nodes = {n for e in es for n in e[:2]} | (keep or set())
    nodes = {n for n in nodes if not n.startswith("ext: ") or n in site_ext or keep is None}
    ids = {n: f"n{i}" for i, n in enumerate(sorted(nodes))}
    for n, i in ids.items():
        if n in site_ext:
            continue  # drawn per call site below
        info = funcs.get(n)
        label = f"{n}\n{info['file']}:{info['node'].lineno if info and info.get('node') else 1}" if info else n
        attrs = {}
        if n.startswith("ext: "):
            attrs = {"shape": "plaintext", "fontcolor": "gray30"}
        elif n in site_funcs:
            attrs = {"style": "filled", "fillcolor": "#ffd6d6", "color": "#d00000", "penwidth": "3", "fontcolor": "#a00000"}
            label = "LLM CALL INSIDE\n" + label
        elif n in targets:
            weak = targets[n] >= 2
            attrs = {"color": "#f0a0a0" if weak else "#d00000", "style": "dashed", "penwidth": "1.5" if weak else "2"}
            label += f"\nname-links: {targets[n]}"
        if n in entries:
            attrs.update({"style": "filled", "fillcolor": "#dbe6ff"})
        g.node(i, label=label.replace("\\", "\\\\"), **attrs)
    for k, s in enumerate(sites):
        if s["func"] not in ids:
            continue
        sid = f"s{k}"
        g.node(sid, label=f"{s['rule']}\n{s['pattern']}\n{s['detail'][:60]}\nL{s['line']}".replace("\\", "\\\\"),
               shape="ellipse", peripheries="2", color="#d00000", penwidth="2.5", fontcolor="#a00000", fontsize="9")
        g.edge(ids[s["func"]], sid, color="#d00000", penwidth="2", label=f"L{s['line']}", fontsize="8")
    for a, b, line, kind in sorted(set(es)):
        if a in ids and b in ids and b not in site_ext:
            g.edge(ids[a], ids[b], label=f"L{line}", fontsize="8", **EDGE_STYLE[kind])
    with g.subgraph(name="cluster_legend") as lg:
        lg.attr(label="legend", fontsize="10", color="gray")
        lg.node("lg1", "LLM call (rule hit)", shape="ellipse", peripheries="2", color="#d00000", fontcolor="#a00000", fontsize="9")
        lg.node("lg2", "contains an LLM call\n(CFG target)", style="filled", fillcolor="#ffd6d6", color="#d00000", penwidth="3", fontsize="9")
        lg.node("lg3", "reaches an LLM call\n(CFG target)", style="dashed", color="#d00000", penwidth="2", fontsize="9")
        lg.node("lg3b", "reaches via 2+ same-name links\n(weaker evidence, check the path)", style="dashed", color="#f0a0a0", fontsize="9")
        lg.node("lg4", "entry", style="filled", fillcolor="#dbe6ff", fontsize="9")
        lg.edge("lg1", "lg2", style="invis"); lg.edge("lg2", "lg3", style="invis")  # noqa: E702
        lg.edge("lg3", "lg3b", style="invis"); lg.edge("lg3b", "lg4", style="invis")  # noqa: E702
    g.render(out / name, format="svg", cleanup=True)
    g.render(out / name, format="png", cleanup=True)


PACKAGE = ""


def main():
    p = argparse.ArgumentParser()
    p.add_argument("src_root", type=Path)
    p.add_argument("out_dir", type=Path)
    p.add_argument("--entry", nargs="*", default=[])
    p.add_argument("--rules", default="R1,R2,R3")
    p.add_argument("--cfg", action="store_true", help="also draw CFGs of the CFG targets")
    p.add_argument("--view-max-links", type=int, default=None,
                   help="callgraph_rules_targets: hide targets needing more name-links than this (view only; the TSV keeps all)")
    p.add_argument("--exclude", nargs="*", default=[])
    p.add_argument("--only", nargs="*", default=[])
    a = p.parse_args()
    global PACKAGE
    PACKAGE = a.src_root.resolve().name
    llm_rules.RULES = set(a.rules.split(","))
    a.out_dir.mkdir(parents=True, exist_ok=True)

    files = audit.sources(a.src_root, a.only)
    funcs, classes = audit.scan(a.src_root, a.only)
    funcs = {q: i for q, i in funcs.items() if not any(x in q for x in a.exclude)}
    imports, cls_nodes = module_info(a.src_root, files)
    sites = find_sites(a.src_root, files, funcs, classes, imports, cls_nodes)
    es = set(audit.edges(funcs, classes)) | bridge_edges(audit.edges(funcs, classes), funcs, imports)

    site_funcs = {s["func"] for s in sites}
    targets = reverse_closure(site_funcs, [e for e in es if not e[1].startswith("ext: ")])
    entries = set(a.entry)

    with open(a.out_dir / "rule_sites.tsv", "w", encoding="utf-8") as fh:
        fh.write("location\tfunction\trule\tpattern\tdetail\n")
        for s in sites:
            fh.write(f"{s['file']}:{s['line']}\t{s['func']}\t{s['rule']}\t{s['pattern']}\t{s['detail']}\n")
    with open(a.out_dir / "cfg_targets.tsv", "w", encoding="utf-8") as fh:
        fh.write("function\tfile\twhy\tname_links\n")
        for q in sorted(targets, key=lambda q: (targets[q], q not in site_funcs, q)):
            why = "contains an LLM call" if q in site_funcs else "reaches an LLM call"
            fh.write(f"{q}\t{funcs[q]['file']}\t{why}\t{targets[q]}\n")

    draw("callgraph_rules_full", a.out_dir, funcs, es, sites, site_funcs, targets, entries)
    shown = {q for q, d in targets.items() if a.view_max_links is None or d <= a.view_max_links}
    draw("callgraph_rules_targets", a.out_dir, funcs, es, sites, site_funcs, targets, entries, keep=shown)
    if a.entry:
        reach = audit.reachable(a.entry, es)
        draw("callgraph_rules_entry", a.out_dir, funcs, [e for e in es if e[0] in reach], sites, site_funcs,
             targets, entries)
        missing = sorted(site_funcs - reach)
        if missing:
            print("WARNING: LLM calls not reachable from --entry:", ", ".join(missing))
    if a.cfg:
        audit.draw_cfgs({q: funcs[q] for q in targets if funcs[q].get("node") is not None}, a.out_dir, [])

    print(f"functions={len(funcs)} llm_calls={len(sites)} cfg_targets={len(targets)} "
          f"(contain={len(site_funcs)}, reach with <=1 name-link={sum(1 for v in targets.values() if v <= 1) - len(site_funcs)}, "
          f"reach only with 2+ name-links={sum(1 for v in targets.values() if v >= 2)})")


if __name__ == "__main__":
    main()

"""Colleague's call graph, unchanged, with our R1-R3 findings marked in red on top.

Same scope, same nodes, same edges and same labels as audit.py (so the picture matches 1_colleague_callgraph);
only colours are added. The LLM calls themselves usually live outside the colleague's scope (models/), so the
whole package is analysed in the background and the result is projected onto the colleague's graph:

  red filled box         function in the graph that contains an LLM call itself
  red bold border        function in the graph that reaches an LLM call (a wrapper may sit here)
  red double ellipse     external call (e.g. `self.model.query`) that leads into an LLM call outside the scope
  light pink dashed      reaches an LLM call only through more than --max-links same-name guesses (weak evidence)
  green double border    function that already logs [LLM_*] (existing wrapper)
  pink dashed box        (in the cluster) raw HTTP whose host could not be resolved: check by hand
  red dashed cluster     (only with --show-outside) OUTSIDE the colleague's scope: the functions that make the LLM calls (red filled boxes)
                         and the calls themselves (red double circles), hung off the red external node they are
                         reached through (dotted edge, "via" = the function the name-link lands on)
  yellow fill            --mark, exactly as in audit.py

usage (repo root):
  python PoC/verification/mark_colleague.py src/minisweagent src/minisweagent/agents <out_dir> \
      --entry DefaultAgent.run --exclude InteractiveAgent --through auto
  python mark_colleague.py src/smolagents src/smolagents <out> --only agents.py tools.py --entry MultiStepAgent.run --through auto
  python mark_colleague.py browser_use browser_use <out> --only agent/service.py tools/registry/service.py \
      --entry Agent.run --through auto

  --through auto  keep only entry -> colleague's wrapper ([LLM_*] log) -> LLM call; other LLM paths stay uncoloured

outputs: callgraph_full_marked.svg/.png, callgraph_entry_marked.svg/.png (with --entry),
         marked_nodes.tsv (why each red node is red)
"""

import argparse
import ast
import sys
from pathlib import Path

import graphviz

sys.path.insert(0, str(Path(__file__).resolve().parent))
import audit  # noqa: E402  (colleague's, unchanged)
import audit_rules  # noqa: E402
import llm_rules  # noqa: E402

LOG_TAG = "[LLM_"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("pkg_root", type=Path, help="whole package, e.g. src/minisweagent (where the LLM calls are)")
    p.add_argument("colleague_root", type=Path, help="the colleague's scope, e.g. src/minisweagent/agents")
    p.add_argument("out_dir", type=Path)
    p.add_argument("--entry", nargs="*", default=[])
    p.add_argument("--mark", nargs="*", default=[])
    p.add_argument("--exclude", nargs="*", default=[])
    p.add_argument("--only", nargs="*", default=[], help="same as audit.py --only (files relative to colleague_root)")
    p.add_argument("--through", nargs="*", default=[],
                   help="limit the red marks to LLM calls made THROUGH these functions (the colleague's wrappers): "
                        "entry -> wrapper -> LLM call. 'auto' = every function in the colleague's graph that logs "
                        "[LLM_*] itself or through a pure log helper (e.g. smolagents' _log_llm_query)")
    p.add_argument("--show-outside", action="store_true",
                   help="also attach, outside the colleague's scope, the functions that make the LLM calls and the calls "
                        "themselves (red dashed box). Default: off = only the colleague's own nodes and edges, coloured")
    p.add_argument("--rules", default="R1,R2,R3")
    p.add_argument("--max-links", type=int, default=1,
                   help="red only when the LLM call is reached with at most this many same-name (guessed) links; "
                        "weaker ones are light pink (default 1 = typical dynamic dispatch like self.model.query)")
    a = p.parse_args()
    llm_rules.RULES = set(a.rules.split(","))
    a.out_dir.mkdir(parents=True, exist_ok=True)

    # 1) the colleague's graph, exactly as audit.py builds it
    funcs_c, classes_c = audit.scan(a.colleague_root, a.only)
    funcs_c = {q: i for q, i in funcs_c.items() if not any(x in q for x in a.exclude)}
    es_c = audit.edges(funcs_c, classes_c)

    # 2) whole-package analysis (audit_rules): LLM calls, who reaches them, name-bridged external calls
    audit_rules.PACKAGE = a.pkg_root.resolve().name
    # audit.py keys classes by name only: if two files define `class Agent`, the last one scanned wins.
    # Scan the colleague's files LAST so their classes are the ones the projection talks about.
    col_files = {f.resolve() for f in audit.sources(a.colleague_root, a.only)}
    all_files = sorted(a.pkg_root.rglob("*.py"))
    order = [f for f in all_files if f.resolve() not in col_files] + [f for f in all_files if f.resolve() in col_files]
    rel_order = [f.relative_to(a.pkg_root).as_posix() for f in order]
    clash = {}
    for f in order:
        for n in ast.parse(f.read_text(encoding="utf-8")).body:
            if isinstance(n, ast.ClassDef):
                clash.setdefault(n.name, []).append(f.relative_to(a.pkg_root).as_posix())
    clash = {k: v for k, v in clash.items() if len(v) > 1 and any((a.pkg_root / x).resolve() in col_files for x in v)}
    if clash:
        print("class-name clashes resolved in favour of the colleague's files: "
              + "; ".join(f"{k} ({', '.join(v)})" for k, v in sorted(clash.items())))
    files_p = audit.sources(a.pkg_root, rel_order)
    funcs_p, classes_p = audit.scan(a.pkg_root, rel_order)
    imports, cls_nodes = audit_rules.module_info(a.pkg_root, files_p)
    sites = audit_rules.find_sites(a.pkg_root, files_p, funcs_p, classes_p, imports, cls_nodes)
    es_p_raw = audit.edges(funcs_p, classes_p)
    bridges = audit_rules.bridge_edges(es_p_raw, funcs_p, imports)
    internal = [e for e in set(es_p_raw) | bridges if not e[1].startswith("ext: ")]
    site_funcs = {s["func"] for s in sites}
    # raw HTTP to a host that is not a known LLM API (R3 "host unknown") is weak evidence: light pink, not red
    strong_sites = {s["func"] for s in sites if not (s["rule"].startswith("R3") and "host unknown" in s["pattern"])}
    reach = audit_rules.reverse_closure(site_funcs, internal)  # function -> name-links needed

    prefix = a.colleague_root.resolve().relative_to(a.pkg_root.resolve()).as_posix()
    to_pkg = lambda q: q if "#" not in q else (f"{prefix}/{q}" if prefix != "." else q)  # noqa: E731

    adj = {}
    for x, y, *_ in internal:
        adj.setdefault(x, set()).add(y)

    def llm_funcs_from(start):
        seen, todo = set(), [start]
        while todo:
            n = todo.pop()
            if n not in seen:
                seen.add(n)
                todo += adj.get(n, ())
        return sorted(seen & site_funcs)

    # [LLM_*] logs in the colleague's graph; a caller of a pure log helper counts as the wrapper
    direct = {q for q, i in funcs_c.items()
              if any(isinstance(x, ast.Constant) and isinstance(x.value, str) and LOG_TAG in x.value
                     for c in i["calls"] for x in ast.walk(c))}
    helpers = {q for q in direct if to_pkg(q) not in reach}  # logs, but reaches no LLM call itself
    via_helper = {x for x, y, _, k in es_c if k != "external" and y in helpers}
    wrappers = (direct - helpers) | via_helper
    logged = direct | via_helper

    # scope: only what goes THROUGH the colleague's wrappers (entry -> wrapper -> LLM call)
    through = set()
    for spec in a.through:
        through |= wrappers if spec == "auto" else {q for q in funcs_c if q == spec or q.endswith(spec)}
    if a.through and not through:
        sys.exit(f"--through matched nothing in the colleague's graph: {a.through}")
    adj_c, radj_c = {}, {}
    for x, y, *_ in es_c:
        adj_c.setdefault(x, set()).add(y)
        radj_c.setdefault(y, set()).add(x)

    def closure(start, nxt):
        seen, todo = set(start), list(start)
        while todo:
            for m in nxt.get(todo.pop(), ()):
                if m not in seen:
                    seen.add(m)
                    todo.append(m)
        return seen

    if through:
        down = closure(through, adj_c)    # wrappers and what they call (inside the colleague's graph)
        up = closure(through, radj_c)     # who calls the wrappers
        in_scope_sites = set()
        for t in through:                 # LLM calls the wrappers can reach in the whole package
            in_scope_sites |= set(llm_funcs_from(to_pkg(t)))
        reach = audit_rules.reverse_closure(in_scope_sites, internal)
        site_funcs_scope = in_scope_sites
        on_path = lambda q: q in up or (q in down and to_pkg(q) in reach)  # noqa: E731
        print(f"scope (--through): {', '.join(sorted(through))} -> {len(in_scope_sites)} LLM-calling functions")
    else:
        down, site_funcs_scope, on_path = set(funcs_c), site_funcs, (lambda q: to_pkg(q) in reach)  # noqa: E731

    # 3) project onto the colleague's nodes
    why = {}
    for q, info in funcs_c.items():
        pq = to_pkg(q)
        if not on_path(q):
            continue
        if pq in site_funcs_scope:
            why[q] = ("contains", ", ".join(f"L{s['line']} {s['rule']} {s['pattern']}" for s in sites if s["func"] == pq))
        elif q in through or (through and q in up):
            why[q] = ("reaches", "the colleague's wrapper" if q in through else "calls the colleague's wrapper")
        elif pq in reach:
            fs = [f for f in llm_funcs_from(pq) if f in site_funcs_scope]
            strong = [f for f in fs if f in strong_sites]
            why[q] = ("reaches" if reach[pq] <= a.max_links and strong else "weak",
                      f"name-links={reach[pq]}; LLM calls: " + ", ".join(fs)
                      + ("" if strong else " (only raw HTTP to an unknown host)"))
    ext_reach = {}  # strong external node -> {LLM-calling function outside the scope: via which function}
    for caller, callee, line, kind in es_c:
        if kind != "external" or callee in why or caller not in down:
            continue
        hit = sorted({t for c, t, l, k in bridges if c == to_pkg(caller) and l == line and t in reach})
        llm_of = lambda t: [f for f in llm_funcs_from(t) if f in site_funcs_scope]  # noqa: E731
        hit = [t for t in hit if llm_of(t)]
        if hit:
            d = 1 + min(reach[t] for t in hit)  # the bridge itself is one name-link
            llm = sorted({f for t in hit for f in llm_of(t)})
            why[callee] = ("external" if d <= a.max_links else "weak-external",
                           f"name-links={d}; from {caller} L{line} -> {', '.join(hit)}; LLM calls: {', '.join(llm)}")
            if d <= a.max_links:
                cls_of = lambda q: q.split("#")[-1].rsplit(".", 1)[0]  # noqa: E731
                for f in llm:  # "via": the landing function of the same class first, then the closest one
                    via = min((t for t in hit if f in llm_of(t)), key=lambda t: (cls_of(t) != cls_of(f), reach[t], t))
                    ext_reach.setdefault(callee, {})[f] = via

    def draw(es, name):  # audit.draw_callgraph, same graph/labels, plus colours
        g = graphviz.Digraph(name, graph_attr={"rankdir": "LR"}, node_attr={"fontsize": "10", "shape": "box"})
        ids = {n: f"n{i}" for i, n in enumerate(sorted({n for e in es for n in e[:2]}))}
        for n, i in ids.items():
            attrs = {"style": "filled", "fillcolor": "#ffe08a"} if any(m in n for m in a.mark) else {}
            if n.startswith("ext: "):
                attrs = {"shape": "plaintext", "fontcolor": "gray30", **attrs}
            w = why.get(n, (None,))[0]
            if w == "contains":
                attrs.update({"style": "filled", "fillcolor": "#ffd6d6", "color": "#d00000", "penwidth": "3", "fontcolor": "#a00000"})
            elif w == "reaches":
                attrs.update({"color": "#d00000", "penwidth": "2.5", "fontcolor": "#a00000"})
            elif w == "external":
                attrs.update({"shape": "ellipse", "peripheries": "2", "color": "#d00000", "penwidth": "2", "fontcolor": "#a00000"})
            elif w in ("weak", "weak-external"):
                attrs.update({"color": "#f0a0a0", "style": "dashed"})
            if n in logged:
                attrs.update({"peripheries": "2", "color": "#1a9e3a" if w is None else attrs.get("color", "#1a9e3a")})
                if w is None:
                    attrs["penwidth"] = "2"
            label = f"{n}\n{funcs_c[n]['file']}:{funcs_c[n]['node'].lineno}" if n in funcs_c else n
            g.node(i, label=label.replace("\\", "\\\\"), **attrs)
        for x, y, line, kind in sorted(set(es)):
            red = why.get(y, (None,))[0] in ("contains", "reaches", "external") and why.get(x, (None,))[0] in ("contains", "reaches")
            extra = {"color": "#d00000", "penwidth": "1.6"} if red else {}
            g.edge(ids[x], ids[y], label=f"L{line}", fontsize="8", **{**audit.STYLE[kind], **extra})
        # outside the colleague's scope: the functions that make the LLM calls, hung off the red external node
        shown = {e: r for e, r in ext_reach.items() if e in ids} if a.show_outside else {}
        if shown:
            with g.subgraph(name="cluster_outside") as c:
                c.attr(label=f"outside the colleague's scope ({a.colleague_root.name}/): where the LLM calls are",
                       fontsize="10", color="#d00000", style="dashed", fontcolor="#a00000")
                done, xid = set(), {}
                for e, fs in shown.items():
                    for f, via in sorted(fs.items()):
                        fid = xid.setdefault(f, f"x{len(xid)}")  # stable ids
                        if f not in done:
                            done.add(f)
                            info = funcs_p.get(f, {})
                            ln = info["node"].lineno if info.get("node") is not None else 1
                            if f in strong_sites:
                                c.node(fid, label=f"LLM CALL INSIDE\n{f}\n{info.get('file', '')}:{ln}",
                                       style="filled", fillcolor="#ffd6d6", color="#d00000", penwidth="3", fontcolor="#a00000")
                            else:  # raw HTTP to a host we could not resolve: may or may not be an LLM API
                                c.node(fid, label=f"HTTP, HOST UNKNOWN\n{f}\n{info.get('file', '')}:{ln}",
                                       style="filled,dashed", fillcolor="#fff0f0", color="#e08080", penwidth="2", fontcolor="#a05050")
                            for k, st in enumerate(x for x in sites if x["func"] == f):
                                sid = f"{fid}_{k}"
                                c.node(sid, label=f"{st['rule']}  L{st['line']}\n{st['pattern']}", shape="ellipse",
                                       peripheries="2", color="#d00000", penwidth="2", fontcolor="#a00000", fontsize="9")
                                c.edge(fid, sid, color="#d00000", penwidth="1.6")
                        g.edge(ids[e], fid, style="dotted", color="#d00000", fontcolor="#a00000", fontsize="8",
                               label="" if via == f else f"via {via}")
        g.render(a.out_dir / name, format="svg", cleanup=True)
        g.render(a.out_dir / name, format="png", cleanup=True)

    draw(es_c, "callgraph_full_marked")
    if a.entry:
        keep = audit.reachable(a.entry, es_c)
        draw([e for e in es_c if e[0] in keep], "callgraph_entry_marked")

    with open(a.out_dir / "marked_nodes.tsv", "w", encoding="utf-8") as fh:
        fh.write("node\tmark\twhy\talready_logs_LLM\n")
        order = {"contains": 0, "external": 1, "reaches": 2, "weak-external": 3, "weak": 4}
        for n, (w, d) in sorted(why.items(), key=lambda kv: (order[kv[1][0]], kv[0])):
            fh.write(f"{n}\t{w}\t{d}\t{'yes' if n in logged else 'no'}\n")
    print(f"colleague graph: {len(funcs_c)} functions | red: contains={sum(1 for v in why.values() if v[0] == 'contains')} "
          f"reaches={sum(1 for v in why.values() if v[0] == 'reaches')} external->LLM={sum(1 for v in why.values() if v[0] == 'external')} "
          f"| light pink (only via {a.max_links + 1}+ guessed links): {sum(1 for v in why.values() if v[0].startswith('weak'))} "
          f"| green ([LLM_*] log): {len(logged)} | LLM-calling functions outside the scope: "
          f"{len({f for r in ext_reach.values() for f in r})}"
          + (" (attached: --show-outside)" if a.show_outside else " (listed in marked_nodes.tsv; --show-outside draws them)"))


if __name__ == "__main__":
    main()

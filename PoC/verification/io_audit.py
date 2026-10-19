# /// script
# requires-python = ">=3.10"
# dependencies = ["astor", "graphviz", "py2cfg"]
# ///
"""Prune the call graph down to I/O: keep only functions that make, or can reach, an LLM call, a tool-call
extraction, a tool execution or a user-input read (rules in io_rules.py), then draw it and open it in the viewer.

  coloured filled box   a function that contains an I/O call (colour = category)
  dashed box            a function that reaches one (label: which categories, how many name-links)
  ellipse               the I/O call itself: category, rule, pattern, line
  blue fill             an --entry function
  blue dotted edge      name-based link for dynamic dispatch (`self.model.query` -> every `query`)

Reuses audit.py (scan, edges, CFGs, viewer) and audit_rules.py (name links, reverse closure) unchanged.

usage:
  uv run PoC/verification/io_audit.py src/minisweagent PoC/verification/out_io --entry DefaultAgent.run
      [--max-links N]   keep functions that need at most N name-links to reach an I/O call (default 1)
      [--e2-depth N]    E2 (re/json/yaml parsing) counts only up to N static calls below an LLM-reaching
                        function (default 2); elsewhere it is listed as not counted
      [--exclude TEXT ...] [--only PATH ...]

outputs (out_dir):
  io_sites.tsv          every rule hit: location, function, category, rule, pattern, detail, counted
  io_functions.tsv      every kept function: own categories, reached categories, name-links, reachable from entry
  callgraph_io.svg      the pruned graph;  cfg/<function>.svg  CFGs of the kept functions
  io_checks.tsv         one row per counted I/O call: covered by a wrapper / bypass path / outside the entry
  index.html            viewer: verification list (filter, review, export) + click a node -> CFG, jump to VS Code

Verdict (with --entry): a wrapper is a function that logs the category (WRAPPERS below): a string tag such as
"[TOOL_" (logger.debug("[TOOL_EXECUTION_START] ...")) or an agent_observatory_kit call (observatory_log.tool_start).
  covered   every call path from the entry to the I/O call passes a wrapper (then check the wrapper's CFG)
  bypass    some path avoids every wrapper; the path is shown
  outside   the I/O call is not reachable from the entry
"""

import argparse
import ast
import sys
from pathlib import Path

import graphviz

sys.path.insert(0, str(Path(__file__).resolve().parent))
import audit  # noqa: E402
import audit_rules  # noqa: E402
import io_rules  # noqa: E402

# category -> (string tag, observatory_log functions). EXTRACT parses the LLM reply, so the LLM wrapper covers it.
# The kit has no input events: user input is reported as the tool action "user_input" and counts as INPUT only.
KIT_LLM, KIT_TOOL, KIT_INPUT_ACTION = (
    {"llm_query", "llm_response", "llm_error"},
    {"tool_start", "tool_end", "tool_error"},
    "user_input",
)
WRAPPERS = {
    "LLM": ("[LLM_", KIT_LLM),
    "EXTRACT": ("[LLM_", KIT_LLM),
    "TOOL": ("[TOOL_", KIT_TOOL),
    "INPUT": ("[INPUT_", set()),
}
ROOT = Path(".")
REF_STYLE = {"style": "dashed", "color": "#00897b"}  # T4 dispatch -> function registered by reference
TAB, NL = "\t", "\n"
COLOR = {"LLM": "#d00000", "EXTRACT": "#7b3fb5", "TOOL": "#d46b00", "INPUT": "#1f8a4c"}
FILL = {"LLM": "#ffd6d6", "EXTRACT": "#eadcf7", "TOOL": "#ffe3c4", "INPUT": "#d3f0de"}
# shown in the viewer, built from the same colours the graph is drawn with
LEGEND = {
    "categories": [
        {"color": COLOR[c], "fill": FILL[c], "label": label}
        for c, label in [
            ("LLM", "LLM 호출"),
            ("EXTRACT", "LLM 응답에서 Tool 정보 추출"),
            ("TOOL", "Tool 실행"),
            ("INPUT", "사용자 입력"),
        ]
    ],
    "edges": [
        {"color": "#000000", "style": "solid", "label": "확실한 호출 (self.x())"},
        {"color": "#000000", "style": "dashed", "label": "하위 클래스 override가 대신 실행될 수 있음"},
        {"color": "#0000ff", "style": "solid", "label": "super().x()"},
        {"color": "#3355aa", "style": "dotted", "label": "이름 연결 (정적으로 추정)"},
        {"color": REF_STYLE["color"], "style": "dashed", "label": "등록 함수 연결 (정적으로 추정)"},
        {"color": COLOR["LLM"], "style": "solid", "label": "함수 → I/O 호출 지점 (범주 색)"},
    ],
}


def find_sites(funcs, classes, imports, cls_nodes):
    sites = []
    for q, info in funcs.items():
        imp = imports[info["file"]]
        strings = audit_rules.class_strings(info["cls"], classes, cls_nodes)
        for c in info["calls"]:
            if hit := io_rules.classify(c, imp, strings):
                sites.append(
                    {
                        "func": q,
                        "file": info["file"],
                        "line": c.lineno,
                        "cat": hit[0],
                        "rule": hit[1],
                        "pattern": hit[2],
                        "detail": hit[3],
                    }
                )
        for n, pat in io_rules.extract_fields(info["node"]):
            sites.append(
                {
                    "func": q,
                    "file": info["file"],
                    "line": n.lineno,
                    "cat": "EXTRACT",
                    "rule": "E1-field",
                    "pattern": pat,
                    "detail": ast.unparse(n),
                }
            )
    uniq = {(s["func"], s["line"], s["rule"], s["pattern"]): s for s in sites}
    return sorted(uniq.values(), key=lambda s: (s["file"], s["line"]))


def registered_edges(sites, funcs):
    """T4 dispatch (`self.tools[name](**args)`) calls functions registered by REFERENCE elsewhere in the same class
    (`self.tools = {"bash": bash}`) or module. Link the dispatch site to every package function referenced as a
    value (not called) there: an over-approximation, like the name links."""
    by_name = {}
    for q in funcs:
        by_name.setdefault(q.split("#")[-1].split(".")[-1], []).append(q)
    out = set()
    for s in (s for s in sites if s["rule"] == "T4-dispatch"):
        cls, file = funcs[s["func"]]["cls"], funcs[s["func"]]["file"]
        scope = [i["node"] for i in funcs.values() if (cls and i["cls"] == cls) or (not cls and i["file"] == file)]
        called = {id(c.func) for n in scope for c in ast.walk(n) if isinstance(c, ast.Call)}
        for n in scope:
            for v in ast.walk(n):
                if isinstance(v, (ast.Name, ast.Attribute)) and id(v) not in called and isinstance(v.ctx, ast.Load):
                    name = v.id if isinstance(v, ast.Name) else v.attr
                    out |= {(s["func"], t, s["line"], "ref") for t in by_name.get(name, []) if t != s["func"]}
    return out


def near_llm(llm_funcs, static_es, depth):
    """Functions that reach an LLM call, plus everything up to `depth` static calls below them (where the
    response is typically parsed: query -> _parse_actions -> parse_regex_actions)."""
    near = set(audit_rules.reverse_closure(llm_funcs, static_es))
    frontier = set(near)
    for _ in range(depth):
        frontier = {b for a, b, *_ in static_es if a in frontier} - near
        near |= frontier
    return near


def logs(c: ast.Call, cat: str) -> bool:
    tag, kit = WRAPPERS[cat]
    first = c.args[0].value if c.args and isinstance(c.args[0], ast.Constant) else None
    name = c.func.attr if isinstance(c.func, ast.Attribute) else getattr(c.func, "id", "")
    if isinstance(first, str) and first.startswith(tag):
        return True
    if name in KIT_TOOL and first == KIT_INPUT_ACTION:
        return cat == "INPUT"
    return name in kit


def wrappers(funcs, cat: str) -> set[str]:
    return {q for q, info in funcs.items() if any(logs(c, cat) for c in info["calls"])}


def bypass_path(entries, target, adj, blocked):
    """Shortest call path entry -> target that enters no blocked function, or None."""
    prev, todo = {e: None for e in entries if e not in blocked}, [e for e in entries if e not in blocked]
    while todo:
        n = todo.pop(0)
        if n == target:
            path = []
            while n is not None:
                path.append(n)
                n = prev[n]
            return path[::-1]
        for m in adj.get(n, ()):
            if m not in prev and m not in blocked:
                prev[m] = n
                todo.append(m)
    return None


def checks(sites, funcs, func_es, entries, from_entry):
    adj, calls_at = {}, {}
    for a, b, line, _ in func_es:
        adj.setdefault(a, set()).add(b)
        calls_at.setdefault(line, set()).add((a, b))
    wraps = {cat: wrappers(funcs, cat) for cat in WRAPPERS}
    # logging moved into helpers (smolagents: self._log_llm_query(...) next to self.model.generate(...)): a function
    # that calls such a helper through a resolved call emits the events too
    static = [(a, b) for a, b, _, kind in func_es if kind in ("static", "super")]
    wraps = {cat: ws | {a for a, b in static if b in ws} for cat, ws in wraps.items()}
    out = []
    for s in sites:
        ws = wraps[s["cat"]]
        if s["func"] not in from_entry:
            status, path, via = "outside", [], []
        elif s["func"] in ws:
            status, path, via = "covered", [], [s["func"]]
        elif called := sorted(w for w in ws if (s["func"], w) in calls_at.get(s["line"], set())):
            status, path, via = "covered", [], called  # the hit is the call INTO a wrapper (self.execute_action)
        elif path := bypass_path(entries, s["func"], adj, ws):
            status, via = "bypass", []
        else:
            status, path = "covered", []
            via = sorted(w for w in ws if w in from_entry and bypass_path([w], s["func"], adj, set()) is not None)
        uri = f"vscode://file/{(ROOT / s['file']).resolve().as_posix()}:{s['line']}"
        out.append(
            {
                "id": f"{s['file']}:{s['line']}:{s['rule']}:{s['pattern']}",
                "cat": s["cat"],
                "rule": s["rule"],
                "pattern": s["pattern"],
                "detail": s["detail"],
                "func": s["func"],
                "file": s["file"],
                "line": s["line"],
                "uri": uri,
                "status": status,
                "path": path,
                "via": via,
            }
        )
    return out


def find_kit(src_root: Path) -> Path | None:
    """agent_observatory_kit installed as <repo>/observatory, found by walking up from the analysed package."""
    return next(
        (d / "observatory" for d in src_root.resolve().parents if (d / "observatory/prompt/task.md").exists()), None
    )


def site_name(s: dict) -> str:
    return f"[{s['cat']}] {s['rule']}  {s['pattern']}  @ {s['file']}:{s['line']}"


def draw(out: Path, funcs, es, sites, own, reach, dist, entries):
    g = graphviz.Digraph("callgraph_io", graph_attr={"rankdir": "LR"}, node_attr={"fontsize": "10", "shape": "box"})
    names = sorted({n for e in es for n in e[:2]} | {site_name(s) for s in sites})
    ids = {n: f"n{i}" for i, n in enumerate(names)}
    for n in names:
        if n in funcs:
            info, cats = funcs[n], own.get(n, [])
            label = f"{n}\n{info['file']}:{info['node'].lineno}"
            if cats:
                attrs = {"style": "filled", "fillcolor": FILL[cats[0]], "color": COLOR[cats[0]], "penwidth": "2.5"}
                label = f"{' + '.join(cats)} INSIDE\n{label}"
            else:
                attrs = {"style": "dashed", "color": "#666666"}
                label += f"\nreaches: {', '.join(c for c in io_rules.CATEGORIES if c in reach.get(n, ()))}"
                label += f"  links: {dist.get(n, 0)}" if dist.get(n, 0) else ""
            if n in entries:
                attrs.update({"style": "filled", "fillcolor": "#dbe6ff"})
            g.node(ids[n], label=label.replace("\\", "\\\\"), **attrs)
    for s in sites:
        n = site_name(s)
        g.node(
            ids[n],
            label=f"{s['cat']}  {s['rule']}\n{s['pattern']}\nL{s['line']}".replace("\\", "\\\\"),
            shape="ellipse",
            color=COLOR[s["cat"]],
            fontcolor=COLOR[s["cat"]],
            fontsize="9",
            penwidth="2",
        )
    for a, b, line, kind in sorted(set(es)):
        if a in ids and b in ids:
            style = {"color": COLOR[kind]} if kind in COLOR else audit_rules.EDGE_STYLE.get(kind, REF_STYLE)
            g.edge(ids[a], ids[b], label=f"L{line}", fontsize="8", **style)
    g.render(out / "callgraph_io", format="svg", cleanup=True)
    return {"svg": (out / "callgraph_io.svg").read_text(encoding="utf-8"), "ids": {i: n for n, i in ids.items()}}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("src_root", type=Path)
    p.add_argument("out_dir", type=Path)
    p.add_argument("--entry", nargs="*", default=[])
    p.add_argument("--max-links", type=int, default=1)
    p.add_argument("--e2-depth", type=int, default=2)
    p.add_argument("--exclude", nargs="*", default=[])
    p.add_argument("--only", nargs="*", default=[])
    a = p.parse_args()
    global ROOT
    ROOT = a.src_root
    audit_rules.PACKAGE = a.src_root.resolve().name
    a.out_dir.mkdir(parents=True, exist_ok=True)

    funcs, classes = audit.scan(a.src_root, a.only)
    funcs = {q: i for q, i in funcs.items() if not any(x in q for x in a.exclude)}
    imports, cls_nodes = audit_rules.module_info(a.src_root, audit.sources(a.src_root, a.only))
    sites = find_sites(funcs, classes, imports, cls_nodes)
    base = audit.edges(funcs, classes)
    es = set(base) | audit_rules.bridge_edges(base, funcs, imports) | registered_edges(sites, funcs)
    func_es = [e for e in es if not e[1].startswith("ext: ")]
    static_es = [e for e in func_es if e[3] != "name"]

    near = near_llm({s["func"] for s in sites if s["cat"] == "LLM"}, static_es, a.e2_depth)
    for s in sites:
        s["counted"] = s["rule"] != "E2-parse" or s["func"] in near
    counted = [s for s in sites if s["counted"]]

    own = {}
    for s in counted:
        own.setdefault(s["func"], [])
        if s["cat"] not in own[s["func"]]:
            own[s["func"]].append(s["cat"])
    dist = audit_rules.reverse_closure(set(own), func_es)
    reach = {}
    for cat in io_rules.CATEGORIES:
        for q in audit_rules.reverse_closure({f for f, cs in own.items() if cat in cs}, func_es):
            reach.setdefault(q, set()).add(cat)
    keep = {q for q, d in dist.items() if d <= a.max_links}
    from_entry = audit.reachable(a.entry, es) if a.entry else set(funcs)
    if a.entry:
        keep &= from_entry
        unreached = sorted(set(own) - from_entry)
        if unreached:
            print(f"note: {len(unreached)} I/O functions are not reachable from --entry (see io_functions.tsv)")

    kept_sites = [s for s in counted if s["func"] in keep]
    graph_es = [e for e in func_es if e[0] in keep and e[1] in keep]
    site_es = [(s["func"], site_name(s), s["line"], s["cat"]) for s in kept_sites]
    graph = draw(a.out_dir, funcs, graph_es + site_es, kept_sites, own, reach, dist, set(a.entry))

    with open(a.out_dir / "io_sites.tsv", "w", encoding="utf-8") as fh:
        fh.write("location\tfunction\tcategory\trule\tpattern\tdetail\tcounted\n")
        for s in sites:
            fh.write(
                f"{s['file']}:{s['line']}\t{s['func']}\t{s['cat']}\t{s['rule']}\t{s['pattern']}\t"
                f"{s['detail'][:120]}\t{s['counted']}\n"
            )
    with open(a.out_dir / "io_functions.tsv", "w", encoding="utf-8") as fh:
        fh.write("function\tfile\town\treaches\tname_links\tkept\tfrom_entry\n")
        for q in sorted(dist, key=lambda q: (dist[q], q)):
            fh.write(
                f"{q}\t{funcs[q]['file']}\t{','.join(own.get(q, []))}\t"
                f"{','.join(c for c in io_rules.CATEGORIES if c in reach.get(q, ()))}\t{dist[q]}\t"
                f"{q in keep}\t{q in from_entry}\n"
            )

    found = checks(counted, funcs, func_es, a.entry, from_entry) if a.entry else []
    with open(a.out_dir / "io_checks.tsv", "w", encoding="utf-8") as fh:
        cols = ["status", "category", "rule", "pattern", "location", "function", "wrappers", "bypass_path"]
        rows = [cols] + [
            [
                c["status"],
                c["cat"],
                c["rule"],
                c["pattern"],
                f"{c['file']}:{c['line']}",
                c["func"],
                ",".join(c["via"]),
                " -> ".join(c["path"]),
            ]
            for c in found
        ]
        fh.writelines(TAB.join(r) + NL for r in rows)

    audit.draw_cfgs({q: funcs[q] for q in keep}, a.out_dir, [])
    request = {"kit_found": find_kit(a.src_root) is not None, "audit_cmd": "uv run --python 3.12 " + " ".join(sys.argv)}
    extra = {"checks": found, "legend": LEGEND, "request": request}
    audit.write_viewer(a.src_root, funcs, graph_es + site_es, {"io": graph}, a.out_dir, extra)
    by_cat = {c: sum(1 for s in kept_sites if s["cat"] == c) for c in io_rules.CATEGORIES}
    print(
        f"functions={len(funcs)} kept={len(keep)} io_calls_shown={by_cat} "
        f"rule_hits={len(sites)} (E2 not counted: {sum(1 for s in sites if not s['counted'])})"
    )
    if found:
        print("checks:", {k: sum(1 for c in found if c["status"] == k) for k in ("covered", "bypass", "outside")})


if __name__ == "__main__":
    main()

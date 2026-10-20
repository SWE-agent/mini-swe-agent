# /// script
# requires-python = ">=3.10"
# dependencies = ["astor", "graphviz", "py2cfg"]
# ///
"""Cross-check flow_audit.py with CodeQL: same sources, uses and log calls (codeql/IoFlow.qll), CodeQL's data flow.

Needs the CodeQL CLI bundle (not on PyPI): https://github.com/github/codeql-action/releases (codeql-bundle-win64.tar.gz),
found via --codeql, $CODEQL, PATH, or tools/codeql next to this repository.

usage (repo root):
  uv run --python 3.12 PoC/verification/run_codeql.py PoC/verification/toy_agent PoC/verification/out_codeql \
      --entry ToyAgent.run --only agent.py llm.py tools.py observatory_log.py

outputs (out_dir):
  codeql_flows.csv   io_flow.ql: one row per (source, use) with the log calls the source reaches
  paths.sarif        io_paths.ql: every flow path source -> use / source -> log call
  codeql.svg, index.html   the paths drawn as one graph, each step labelled with its function, in the shared viewer;
                           click a node, "여기서 기록됨 확인": it and what is reachable only through it disappear
                           (--compact: draw each flow only up to its first log call instead)

The database is built in the system temp folder: on this machine Windows refuses CodeQL's final directory rename
under Developments/ (AccessDenied), not under %TEMP%. The Python extractor is pointed at `python` (uv's interpreter)
because it otherwise requires the Windows `py` launcher.
"""

import argparse
import csv
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import graphviz

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import audit  # noqa: E402
import flow_audit  # noqa: E402

LEGEND = {
    **flow_audit.LEGEND,
    "edges": [
        {"color": "#000000", "style": "solid", "label": "데이터 흐름 (CodeQL 경로의 한 단계)"},
        {"color": "#2e7d32", "style": "dashed", "label": "여기서 값이 로그로 감 (… N단계 생략)"},
        {"color": "#888888", "style": "dotted", "label": "--compact: 기록된 뒤 사용 지점까지 (… N단계 생략)"},
    ],
}


def find_codeql(given: str | None) -> str:
    candidates = [given, os.environ.get("CODEQL"), shutil.which("codeql")]
    candidates += [str(p / "tools/codeql" / exe) for p in HERE.parents for exe in ("codeql.exe", "codeql")]
    if found := next((c for c in candidates if c and Path(c).exists()), None):
        return found
    raise SystemExit("CodeQL CLI not found: pass --codeql, set $CODEQL, or unpack the bundle into ../tools/codeql")


def run(cmd: list, **kwargs) -> subprocess.CompletedProcess:
    result = subprocess.run(
        [str(c) for c in cmd], capture_output=True, text=True, encoding="utf-8", errors="replace", **kwargs
    )
    if result.returncode:
        raise SystemExit(f"failed: {' '.join(map(str, cmd[:4]))} ...\n{result.stdout[-2000:]}{result.stderr[-2000:]}")
    return result


def enclosing(funcs: dict) -> callable:
    """(file, line) -> qualname of the innermost function containing the line."""
    spans = [(i["file"], i["node"].lineno, i["node"].end_lineno, q) for q, i in funcs.items()]

    def find(file: str, line: int) -> str:
        hits = [(e - s, q) for f, s, e, q in spans if f == file and s <= line <= e]
        return min(hits)[1] if hits else "<module>"

    return find


def code_at(src: Path, width: int = 70):
    """(file, line) -> that source line, stripped (log and use nodes show it: CodeQL only names the argument)."""
    cache = {}

    def get(file: str, line: int) -> str:
        if file not in cache:
            cache[file] = (src / file).read_text(encoding="utf-8").splitlines()
        return cache[file][line - 1].strip()[:width]

    return get


def read_paths(sarif: Path, func_of, cat_at: dict) -> tuple[list, dict]:
    """SARIF code flows -> paths [(cat, kind, [node, ...])], node -> {file, line, func, text}.
    SARIF merges results that end at the same node (messages joined by newlines), so the category comes from the
    path's first step (cat_at: "file:line" -> category, from the table query) and the end kind from the message."""
    nodes, paths = {}, []
    for result in json.loads(sarif.read_text(encoding="utf-8"))["runs"][0].get("results", []):
        kind = result["message"]["text"].splitlines()[0].split("|")[1]
        for flow in result.get("codeFlows", []):
            for thread in flow["threadFlows"]:
                steps = []
                for step in thread["locations"]:
                    loc = step["location"]
                    file = loc["physicalLocation"]["artifactLocation"]["uri"]
                    line = loc["physicalLocation"]["region"]["startLine"]
                    text = loc.get("message", {}).get("text", "").replace("ControlFlowNode for ", "")
                    name = f"{text} @ {file}:{line}"
                    nodes.setdefault(name, {"file": file, "line": line, "func": func_of(file, line), "text": text})
                    if not steps or steps[-1] != name:
                        steps.append(name)
                first = nodes[steps[0]]
                paths.append((cat_at[f"{first['file']}:{first['line']}"], kind, steps))
    return paths, nodes


def shortest_uses(paths) -> dict:
    """source -> {(use kind, use node): the shortest path}"""
    out = {}
    for cat, kind, steps in paths:
        if kind != "log":
            best = out.setdefault(steps[0], {})
            if (kind, steps[-1]) not in best or len(steps) < len(best[(kind, steps[-1])]):
                best[(kind, steps[-1])] = steps
    return out


def layout(paths, full: bool) -> set:
    """Edges to draw as (a, b, style, skipped steps); style: flow | log | skip.
    Compact (default): per (source, use), only the shortest path up to where the value branches off to a log call
    (the earliest such point); from there one dashed edge to the log and one dotted edge to the use, the steps in
    between counted, not drawn. A use reached without any log is drawn in full: that is what needs checking."""
    if full:
        return {
            (a, b, "log" if kind == "log" and b == steps[-1] else "flow", 0)
            for _, kind, steps in paths
            for a, b in zip(steps, steps[1:])
        }
    logs = {}
    for _, kind, steps in paths:
        if kind == "log":
            logs.setdefault(steps[0], []).append(steps)
    edges = set()
    for src, uses in shortest_uses(paths).items():
        for steps in uses.values():
            branches = []
            for q in logs.get(src, []):
                j = next((k for k, (x, y) in enumerate(zip(steps, q)) if x != y), min(len(steps), len(q))) - 1
                branches.append((j, len(q), q))
            if not branches:
                edges |= {(a, b, "flow", 0) for a, b in zip(steps, steps[1:])}
                continue
            j, _, q = min(branches)
            edges |= {(a, b, "flow", 0) for a, b in zip(steps[: j + 1], steps[1 : j + 1])}
            edges.add((steps[j], q[-1], "log", len(q) - j - 2))
            edges.add((steps[j], steps[-1], "skip", len(steps) - j - 2))
    return edges


def draw(out: Path, paths, nodes, code, full: bool):
    kinds = {}
    for cat, kind, steps in paths:
        kinds[steps[0]] = ("source", cat)
        kinds.setdefault(steps[-1], ("log" if kind == "log" else "use", kind))
    edges = layout(paths, full)
    shown = {n for a, b, _, _ in edges for n in (a, b)}
    font = "Malgun Gothic"
    g = graphviz.Digraph(
        "codeql",
        graph_attr={"rankdir": "LR", "fontname": font},
        node_attr={"fontsize": "10", "fontname": font},
        edge_attr={"fontname": font},
    )
    ids = {n: f"n{i}" for i, n in enumerate(sorted(shown))}
    for n, i in ids.items():
        info, (kind, what) = nodes[n], kinds.get(n, ("step", ""))
        where = f"{info['func']} : L{info['line']}"
        if kind == "source":
            label = f"[{flow_audit.SOURCE[what]}] {info['text']}\n{where}"
            g.node(
                i,
                label,
                shape="ellipse",
                style="filled",
                fillcolor=flow_audit.FILL[what],
                color=flow_audit.COLOR[what],
                penwidth="2",
            )
        elif kind == "use":
            line = code(info["file"], info["line"]).replace("\\", "\\\\")
            g.node(i, f"[{what}] {info['text']}\n{line}\n{where}", shape="doubleoctagon", color="#333333")
        elif kind == "log":
            line = code(info["file"], info["line"]).replace("\\", "\\\\")
            g.node(i, f"[LOG] {line}\n{where}", shape="note", style="filled", fillcolor="#e6f4ea", color="#2e7d32")
        else:
            g.node(i, f"{info['text']}\n{where}", shape="box", style="rounded")
    style = {
        "flow": {},
        "log": {"color": "#2e7d32", "style": "dashed"},
        "skip": {"color": "#888888", "style": "dotted"},
    }
    for a, b, kind, skipped in sorted(edges):
        label = f"… {skipped}단계" if skipped > 0 else ""
        g.edge(ids[a], ids[b], label=label, fontsize="8", **style[kind])
    g.render(out / "codeql", format="svg", cleanup=True)
    return (
        {"svg": (out / "codeql.svg").read_text(encoding="utf-8"), "ids": {i: n for n, i in ids.items()}},
        kinds,
        edges,
    )


def checks(paths, nodes, rows, src_root: Path):
    """One row per (source, use); logged when the source also reaches a log call (CodeQL's verdict, io_flow.ql)."""
    logs, uses = {}, {}
    for cat, kind, steps in paths:
        (logs if kind == "log" else uses).setdefault(steps[0], []).append((cat, kind, steps))
    out = []
    for src, items in sorted(uses.items()):
        via = sorted({f"{s[-1]}  ({nodes[s[-1]]['func']})" for _, _, s in logs.get(src, [])})
        shortest = {}  # several paths per (source, use): keep the shortest
        for cat, kind, steps in items:
            key = (kind, steps[-1])
            if key not in shortest or len(steps) < len(shortest[key][2]):
                shortest[key] = (cat, kind, steps)
        for cat, kind, steps in shortest.values():
            out.append((src, cat, kind, steps, "covered" if via else "bypass", via))
    for r in rows:  # sources with no use at all (from the table query)
        if r["use"] == "(no use)":
            out.append(
                (f"{r['col1'].replace('ControlFlowNode for ', '')} @ {r['col2']}", r["cat"], "", [], "outside", [])
            )
    result = []
    for src, cat, kind, steps, status, via in out:
        file, line = src.rsplit(" @ ", 1)[1].rsplit(":", 1)
        at = lambda n: {"file": nodes[n]["file"], "line": nodes[n]["line"]}  # noqa: E731
        result.append(
            {
                "id": f"{src} -> {kind}:{steps[-1] if steps else ''}",
                "cat": cat,
                "rule": kind or "사용 지점 없음",
                "pattern": src.split(" @ ")[0],
                "detail": "",
                "func": src,
                "file": file,
                "line": int(line),
                "uri": f"vscode://file/{(src_root / file).resolve().as_posix()}:{line}",
                "status": status,
                "path": steps,
                "via": via,
                # the three questions: where the value is created, where a log records it, where it is used
                "src_at": {"file": file, "line": int(line)},
                "logs_at": sorted({(nodes[s[-1]]["file"], nodes[s[-1]]["line"]) for _, _, s in logs.get(src, [])})
                if steps
                else [],
                "use_at": at(steps[-1]) if steps else None,
                "use_kind": kind,
                # everything this flow touches (viewer): the use path and the source's paths to log calls
                "nodes": sorted(set(steps) | {n for _, _, s in logs.get(src, []) for n in s}) if steps else [],
            }
        )
        result[-1]["logs_at"] = [{"file": f, "line": n} for f, n in result[-1]["logs_at"]]
    return result


def write_viewer(out: Path, src_root: Path, graph, nodes, edges, found, code: dict):
    view_nodes, rel = {}, {}
    for n, info in nodes.items():
        uri = f"vscode://file/{(src_root / info['file']).resolve().as_posix()}:{info['line']}"
        cfg = out / "cfg" / f"{audit.cfg_name(info['func'])}.svg"
        view_nodes[n] = {
            "file": info["file"],
            "line": info["line"],
            "end": info["line"],
            "uri": uri,
            "cfg": cfg.relative_to(out).as_posix() if cfg.exists() else None,
            "calls": [],
        }
    for a, b, *_ in sorted(edges):
        rel.setdefault(a, {"in": [], "out": []})["out"].append({"name": b, "line": nodes[b]["line"], "kind": "flow"})
        rel.setdefault(b, {"in": [], "out": []})["in"].append(
            {"name": a, "line": nodes[a]["line"], "kind": "flow", "file": nodes[a]["file"], "uri": view_nodes[a]["uri"]}
        )
    labels = {
        "items": "CodeQL 값의 흐름 (출발점 → 사용 지점)",
        "status": {"bypass": "기록 안 됨", "covered": "기록됨", "outside": "사용 안 됨"},
        "cat": flow_audit.SOURCE,
        "kind": {"flow": "데이터 흐름"},
        "tab": {"codeql": "CodeQL 흐름"},
        "sub": {
            "bypass": "로그 없이 흐른 경로",
            "covered": "이 값이 닿는 로그 (함수)",
            "outside": "어떤 사용 지점에도 닿지 않음",
        },
    }
    data = {
        "root": src_root.resolve().as_posix(),
        "graphs": {"codeql": graph},
        "nodes": view_nodes,
        "rel": rel,
        "checks": found,
        "legend": LEGEND,
        "labels": labels,
        "prune": True,
        "cards": True,
        "code": code,
    }
    html = (HERE / "viewer.html").read_text(encoding="utf-8")
    (out / "index.html").write_text(html.replace("__DATA__", json.dumps(data).replace("</", r"<\/")), encoding="utf-8")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("src_root", type=Path)
    p.add_argument("out_dir", type=Path)
    p.add_argument("--entry", required=True, help="Class.method")
    p.add_argument("--only", nargs="*", default=[], help="files to analyse, relative to src_root (default: every *.py)")
    p.add_argument("--codeql")
    p.add_argument("--compact", action="store_true", help="draw each flow only up to its first log call")
    a = p.parse_args()

    codeql = find_codeql(a.codeql)
    work = Path(tempfile.gettempdir()) / "io_codeql" / a.src_root.resolve().name
    shutil.rmtree(work, ignore_errors=True)
    src, pack, db = work / "src", work / "pack", work / "db"
    rels = a.only or [
        f.relative_to(a.src_root).as_posix() for f in a.src_root.rglob("*.py") if "__pycache__" not in f.parts
    ]
    for r in rels:
        (src / r).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(a.src_root / r, src / r)
    shutil.copytree(HERE / "codeql", pack)
    cls, method = a.entry.rsplit(".", 1)
    (pack / "Entry.qll").write_text(
        f'predicate entry(string cls, string method) {{ cls = "{cls}" and method = "{method}" }}\n'
    )

    env = {**os.environ, "PATH": str(Path(sys.executable).parent) + os.pathsep + os.environ["PATH"]}
    print(f"codeql: {codeql}\ndatabase: {db} ({len(rels)} files)")
    run(
        [
            codeql,
            "database",
            "create",
            db,
            "--language=python",
            f"--source-root={src}",
            "--extractor-option=python.python_executable_name=python",
        ],
        env=env,
    )
    run([codeql, "pack", "install", pack])
    a.out_dir.mkdir(parents=True, exist_ok=True)
    run([codeql, "query", "run", pack / "io_flow.ql", "-d", db, "-o", work / "flows.bqrs"])
    run([codeql, "bqrs", "decode", "--format=csv", f"--output={a.out_dir / 'codeql_flows.csv'}", work / "flows.bqrs"])
    print(run([codeql, "bqrs", "decode", "--format=text", work / "flows.bqrs"]).stdout)
    run(
        [
            codeql,
            "database",
            "analyze",
            db,
            pack / "io_paths.ql",
            "--format=sarif-latest",
            f"--output={a.out_dir / 'paths.sarif'}",
            "--rerun",
        ]
    )

    funcs, _ = audit.scan(src, [])
    rows = list(csv.DictReader(open(a.out_dir / "codeql_flows.csv", encoding="utf-8")))
    paths, nodes = read_paths(a.out_dir / "paths.sarif", enclosing(funcs), {r["col2"]: r["cat"] for r in rows})
    graph, _, edges = draw(a.out_dir, paths, nodes, code_at(src), not a.compact)
    audit.draw_cfgs({q: funcs[q] for q in {n["func"] for n in nodes.values()} if q in funcs}, a.out_dir, [])
    found = checks(paths, nodes, rows, a.src_root)
    line_of = code_at(src, 200)
    code = {
        f"{loc['file']}:{loc['line']}": line_of(loc["file"], loc["line"])
        for c in found
        for loc in [c["src_at"], *c["logs_at"], c["use_at"]]
        if loc
    }
    write_viewer(a.out_dir, a.src_root, graph, nodes, edges, found, code)
    print(
        f"paths={len(paths)} steps={len(nodes)} checks="
        f"{ {k: sum(1 for c in found if c['status'] == k) for k in ('covered', 'bypass', 'outside')} }"
        f"\nviewer: {a.out_dir / 'index.html'}"
    )


if __name__ == "__main__":
    main()

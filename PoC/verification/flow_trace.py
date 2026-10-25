# /// script
# requires-python = ">=3.12"
# dependencies = ["anthropic"]
# ///
"""Follow the data a wrapper hands out (LLM reply, tool output, user input) through the agent and check that it is
logged before it leaves the agent. Draws the flow as a graph.

Every source value carries a canary token (CANARY_<LLM|TOOL|USER>_<n>). sys.monitoring watches the agent's own code:
  arg      a function received the canary in its arguments           (caller -> callee)
  return   a function returned it                                     (callee -> caller)
  raise    an exception carried it                                    (raiser -> handler)
  call     it was passed to a sink (subprocess.run, exec, input, ...)  (function -> [sink])
  state    it showed up in a function that did not receive it directly (stored on an object, read later)
  source   where it entered: [LLM API], [process output], [stdin], [entry args]
Calls into observatory_log are the logs. A log call containing the canary logs it. llm_query/tool_start made while the
function holds an unlogged canary in its arguments (llm_response/tool_end: in what it got back) without it "omits" it.
A sink reached before any log contained the canary is a gap. Every LLM query is also checked backwards: text in it that
is neither a source value nor a string constant of the agent's code is "unexplained" (an input no wrapper saw).

usage (from this folder):  uv run flow_trace.py toy [out_dir] [--hidden]
  --hidden  appends the working directory to the toy agent's system prompt, to show an unexplained query input
Writes <out_dir>/flow.html (graph rendered in the browser), flow.dot and flow.json. Default out_dir: out_flow.
"""

import builtins
import html
import io
import json
import os
import re
import subprocess
import sys
import types
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace as NS

CANARY = re.compile(r"CANARY_(?:LLM|TOOL|USER)_\d+")
COLOR = {"LLM": "#2563eb", "TOOL": "#ea580c", "USER": "#16a34a"}
ORIGIN = {"LLM": "[LLM API]", "TOOL": "[process output]", "USER": "[stdin]"}
LOG_SIDE = {"llm_query": "in", "tool_start": "in", "llm_response": "out", "tool_end": "out", "llm_error": None}
LOG_SIDE["tool_error"] = None
SKIP = (type, types.ModuleType, types.FunctionType, types.MethodType, types.BuiltinFunctionType, types.CodeType)
M, E = sys.monitoring, sys.monitoring.events
TOOL_ID = M.DEBUGGER_ID


def strings(value, depth: int = 0, seen: set | None = None) -> list[str]:
    seen = set() if seen is None else seen
    if isinstance(value, str):
        return [value]
    if depth > 12 or id(value) in seen or isinstance(value, SKIP):
        return []
    seen.add(id(value))
    if isinstance(value, dict):
        items = list(value.values())
    elif isinstance(value, (list, tuple, set, frozenset)):
        items = list(value)
    elif isinstance(value, BaseException):
        items = [value.args, vars(value)]
    elif hasattr(value, "__dict__"):
        items = [vars(value)]
    else:
        return []
    return [s for v in items for s in strings(v, depth + 1, seen)]


def tokens(value) -> set[str]:
    return {m.group(0) for s in strings(value) for m in CANARY.finditer(s)}


def kind(canary: str) -> str:
    return canary.split("_")[1]


def short(canaries) -> str:
    return ", ".join(c.removeprefix("CANARY_") for c in sorted(canaries))


def code_strings(const):
    if isinstance(const, str):
        yield const
    elif isinstance(const, (tuple, frozenset)):
        for c in const:
            yield from code_strings(c)
    elif isinstance(const, types.CodeType):
        yield from code_strings(const.co_consts)


class Tracer:
    def __init__(self, root: Path, boundaries: dict[types.CodeType, str], sinks: list[tuple]):
        self.root, self.boundaries, self.sinks = str(root), boundaries, sinks
        self.edges = defaultdict(lambda: {"canaries": set(), "unlogged": set()})
        self.holders: dict[str, list[str]] = defaultdict(list)
        self.logged: dict[str, list[tuple[str, str]]] = defaultdict(list)
        self.omits: dict[tuple[str, str], set[str]] = defaultdict(set)
        self.frames: dict[int, dict[str, set]] = {}
        self.exc_origin: dict[int, tuple[BaseException, str]] = {}
        self.first: set[str] = set()
        self.known: set[str] = set()
        self.queries: list[list[str]] = []
        self.sink_hits: list[tuple[str, str, str, bool]] = []

    def __enter__(self):
        M.use_tool_id(TOOL_ID, "flow_trace")
        for event, callback in [
            (E.PY_START, self._start),
            (E.PY_RETURN, self._return),
            (E.PY_UNWIND, self._unwind),
            (E.CALL, self._call),
            (E.RAISE, self._raise),
            (E.EXCEPTION_HANDLED, self._handled),
        ]:
            M.register_callback(TOOL_ID, event, callback)
        M.set_events(TOOL_ID, E.PY_START | E.PY_RETURN | E.PY_UNWIND | E.CALL | E.RAISE | E.EXCEPTION_HANDLED)
        return self

    def __exit__(self, *_):
        M.set_events(TOOL_ID, 0)
        M.free_tool_id(TOOL_ID)

    def node(self, code: types.CodeType) -> str:
        return self.boundaries.get(code) or f"{Path(code.co_filename).stem}.{code.co_qualname}"

    def is_log(self, code: types.CodeType) -> bool:
        return code.co_filename.startswith(self.root) and Path(code.co_filename).stem == "observatory_log"

    def traced(self, code: types.CodeType) -> bool:
        if code in self.boundaries:
            return True
        return code.co_filename.startswith(self.root) and not code.co_name.startswith("<") and not self.is_log(code)

    def caller(self, frame, missing: str) -> str:
        return self.node(frame.f_back.f_code) if frame.f_back and self.traced(frame.f_back.f_code) else missing

    def see(self, value, toks: set[str]) -> None:
        """Remember the source text of canaries seen for the first time (for the backward query check)."""
        for c in toks - self.first:
            self.known.update(s for s in strings(value) if c in s)
        self.first |= toks

    def writer(self, c: str, frame) -> str:
        """The innermost function on the stack that held `c`: where a value read back from state was stored."""
        f = frame.f_back if frame else None
        while f:
            if self.traced(f.f_code) and self.node(f.f_code) in self.holders[c]:
                return self.node(f.f_code)
            f = f.f_back
        return self.holders[c][-1]

    def flow(self, c: str, src: str, dst: str, label: str, frame=None) -> None:
        """Record that canary `c` moved from `src` to `dst`. `frame` is src's frame (to explain how src got it)."""
        if src not in self.holders[c]:
            if not src.startswith("[") and not self.holders[c]:
                self.edges[(ORIGIN[kind(c)], src, "source", kind(c))]["canaries"].add(c)
            elif not src.startswith("["):
                obj = frame.f_locals.get("self") if frame else None
                attr = next((k for k, v in getattr(obj, "__dict__", {}).items() if c in tokens(v)), None)
                state = f"state (self.{attr})" if attr else "state"
                self.edges[(self.writer(c, frame), src, state, kind(c))]["canaries"].add(c)
            self.holders[c].append(src)
        edge = self.edges[(src, dst, label, kind(c))]
        edge["canaries"].add(c)
        if dst.startswith("["):
            if not self.logged[c]:
                edge["unlogged"].add(c)
            self.sink_hits.append((c, src, dst, bool(self.logged[c])))
        elif dst not in self.holders[c]:
            self.holders[c].append(dst)

    def _start(self, code, offset):
        if self.is_log(code):
            if code.co_name not in LOG_SIDE:
                return M.DISABLE
            frame = sys._getframe(1)
            args = dict(frame.f_locals)
            toks, where = tokens(args), self.node(frame.f_back.f_code)
            self.see(args, toks)
            if side := LOG_SIDE[code.co_name]:
                held = self.frames.get(id(frame.f_back), {}).get(side, set())
                self.omits[(where, code.co_name)].update(c for c in held - toks if not self.logged[c])
            for c in toks:
                self.logged[c].append((where, code.co_name))
            return None
        if not self.traced(code):
            return M.DISABLE
        frame = sys._getframe(1)
        args = {k: v for k, v in frame.f_locals.items() if k not in ("self", "cls")}
        toks = tokens(args)
        self.see(args, toks)
        if code in self.boundaries:
            self.queries.append(strings(args))
        for c in toks:
            self.flow(c, self.caller(frame, "[entry args]"), self.node(code), "arg", frame.f_back)
        self.frames[id(frame)] = {"in": toks, "out": set()}
        return None

    def _return(self, code, offset, value):
        if not self.traced(code):
            return M.DISABLE
        frame = sys._getframe(1)
        toks = tokens(value)
        if code in self.boundaries:
            self.known.update(strings(value))
        self.see(value, toks)
        for c in toks:
            self.flow(c, self.node(code), self.caller(frame, "[return to caller]"), "return", frame)
        if (f := self.frames.get(id(frame.f_back))) is not None:
            f["out"] |= toks
        self.frames.pop(id(frame), None)
        return None

    def _unwind(self, code, offset, exc):
        self.frames.pop(id(sys._getframe(1)), None)

    def _call(self, code, offset, fn, arg0):
        if not self.traced(code):
            return M.DISABLE
        sink = next((name for f, name in self.sinks if f is fn), None)
        if sink and arg0 is not M.MISSING:
            for c in tokens(arg0):
                self.flow(c, self.node(code), sink, "call", sys._getframe(1))
        return None

    def _raise(self, code, offset, exc):
        if self.traced(code) and id(exc) not in self.exc_origin:
            self.exc_origin[id(exc)] = (exc, self.node(code))

    def _handled(self, code, offset, exc):
        origin = self.exc_origin.get(id(exc))
        if not self.traced(code) or not origin or origin[1] == self.node(code):
            return
        toks = tokens(exc)
        for c in toks:
            self.flow(c, origin[1], self.node(code), "raise")
        if (f := self.frames.get(id(sys._getframe(1)))) is not None:
            f["out"] |= toks

    def unexplained(self) -> list[dict]:
        files = {
            Path(m.__file__)
            for m in list(sys.modules.values())
            if (getattr(m, "__file__", None) or "").startswith(self.root)
        }
        consts = {s for f in files for s in code_strings(compile(f.read_text(encoding="utf-8"), str(f), "exec"))}
        pieces = sorted({s for s in self.known | consts if len(s) >= 4}, key=len, reverse=True)
        found = []
        for n, leaves in enumerate(self.queries, 1):
            for leaf in dict.fromkeys(leaves):
                if leaf in self.known or leaf in consts:
                    continue
                rest = leaf
                for piece in pieces:
                    rest = rest.replace(piece, " ")
                if re.search(r"\w{3,}", rest):
                    found.append({"query": n, "text": leaf, "unexplained": " ".join(rest.split())})
        return found

    def report(self) -> dict:
        canaries = sorted(set(self.holders) | set(self.logged) | {h[0] for h in self.sink_hits})
        return {
            "canaries": [
                {
                    "canary": c,
                    "kind": kind(c),
                    "path": self.holders[c],
                    "logged_by": [f"{w}: {fn}" for w, fn in dict.fromkeys(self.logged[c])],
                    "sinks": [
                        {"from": s, "sink": d, "logged_before": ok}
                        for s, d, ok in dict.fromkeys((s, d, ok) for x, s, d, ok in self.sink_hits if x == c)
                    ],
                }
                for c in canaries
            ],
            "gaps": {
                "never_logged": [c for c in canaries if not self.logged[c]],
                "unlogged_sinks": [
                    {"canary": c, "from": s, "sink": d} for c, s, d, ok in dict.fromkeys(self.sink_hits) if not ok
                ],
                "omits": [{"node": w, "log": fn, "canaries": sorted(cs)} for (w, fn), cs in self.omits.items() if cs],
                "unexplained_query_text": self.unexplained(),
            },
        }

    def dot(self) -> str:
        logs = defaultdict(lambda: defaultdict(set))
        for c, entries in self.logged.items():
            for where, fn in entries:
                logs[where][fn].add(c)
        omits = defaultdict(list)
        for (where, fn), cs in self.omits.items():
            if cs:
                omits[where].append(f"⚠ {fn} omits {short(cs)}")
        lines = [
            "digraph flow {",
            '  rankdir=LR; nodesep=0.35; ranksep=0.6; node [fontname="Helvetica" fontsize=10 shape=box style="rounded,filled" fillcolor=white];',
            '  edge [fontname="Helvetica" fontsize=9];',
        ]
        q = lambda s: json.dumps(s, ensure_ascii=False)  # noqa: E731
        for n in sorted({x for s, d, _, _ in self.edges for x in (s, d)}):
            if n.startswith("["):
                lines.append(f'  {q(n)} [shape=ellipse fillcolor="#f1f5f9" color="#94a3b8"];')
                continue
            notes = [f"✓ {fn}: {short(cs)}" for fn, cs in logs[n].items()] + omits[n]
            fill = "#fee2e2" if omits[n] else "#dcfce7" if logs[n] else "white"
            lines.append(f'  {q(n)} [label={q(chr(10).join([n, *notes]))} fillcolor="{fill}"];')
        merged = defaultdict(lambda: {"canaries": set(), "unlogged": set()})
        for (s, d, label, _), e in self.edges.items():
            merged[(s, d, label)]["canaries"] |= e["canaries"]
            merged[(s, d, label)]["unlogged"] |= e["unlogged"]
        for (s, d, label), e in sorted(merged.items()):
            kinds = sorted({kind(c) for c in e["canaries"]})
            attrs = {
                "color": ":".join(COLOR[k] for k in kinds),
                "fontcolor": COLOR[kinds[0]] if len(kinds) == 1 else "#475569",
                "label": f"{label}: {short(e['canaries'])}",
            }
            if label.startswith(("state", "source")):
                attrs["style"] = "dashed"
            if e["unlogged"]:
                attrs |= {"color": "#dc2626", "fontcolor": "#dc2626", "penwidth": "2.5"}
                attrs["label"] = f"⚠ {label}: {short(e['canaries'])}\nunlogged: {short(e['unlogged'])}"
            lines.append(f"  {q(s)} -> {q(d)} [{' '.join(f'{a}={q(v)}' for a, v in attrs.items())}];")
        return "\n".join([*lines, "}"])


def table(head: list[str], rows: list[list]) -> str:
    if not rows:
        return "<p class='none'>none</p>"
    cells = "".join("<tr>" + "".join(f"<td>{html.escape(str(c))}</td>" for c in r) + "</tr>" for r in rows)
    return f"<table><tr>{''.join(f'<th>{h}</th>' for h in head)}</tr>{cells}</table>"


def page(title: str, dot: str, rep: dict) -> str:
    gaps = rep["gaps"]
    sections = [
        (
            "Sinks reached before any log",
            table(["canary", "from", "sink"], [[g["canary"], g["from"], g["sink"]] for g in gaps["unlogged_sinks"]]),
        ),
        (
            "Log calls that omit data the function holds",
            table(["function", "log", "omits"], [[g["node"], g["log"], short(g["canaries"])] for g in gaps["omits"]]),
        ),
        ("Never logged", table(["canary"], [[c] for c in gaps["never_logged"]])),
        (
            "Unexplained LLM query text (no wrapper saw it)",
            table(
                ["query #", "unexplained", "in"],
                [[g["query"], g["unexplained"], g["text"][:160]] for g in gaps["unexplained_query_text"]],
            ),
        ),
        (
            "Canaries",
            table(
                ["canary", "path", "logged by", "sinks"],
                [
                    [
                        c["canary"],
                        " → ".join(c["path"]),
                        "; ".join(c["logged_by"]) or "—",
                        "; ".join(f"{s['sink']} {'✓' if s['logged_before'] else '⚠'}" for s in c["sinks"]),
                    ]
                    for c in rep["canaries"]
                ],
            ),
        ),
    ]
    body = "".join(f"<h2>{h}</h2>{t}" for h, t in sections)
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>{html.escape(title)}</title>
<style>
body {{ font: 14px/1.5 -apple-system, "Segoe UI", Helvetica, sans-serif; margin: 24px; color: #0f172a; background: #fff; }}
h1 {{ font-size: 20px; margin: 0 0 4px; }} h2 {{ font-size: 15px; margin: 28px 0 8px; }}
.legend span {{ margin-right: 16px; }} .none {{ color: #64748b; }}
#graph {{ overflow: auto; border: 1px solid #e2e8f0; border-radius: 8px; padding: 12px; margin-top: 12px; }}
#graph svg {{ max-width: 100%; height: auto; }}
table {{ border-collapse: collapse; width: 100%; }} th, td {{ text-align: left; padding: 4px 8px; border-bottom: 1px solid #e2e8f0; vertical-align: top; }}
th {{ background: #f8fafc; }}
</style></head><body>
<h1>{html.escape(title)}</h1>
<div class="legend"><span style="color:{COLOR["LLM"]}">■ LLM reply</span><span style="color:{COLOR["TOOL"]}">■ tool output</span>
<span style="color:{COLOR["USER"]}">■ user input</span><span style="color:#dc2626">■ reached a sink unlogged</span>
<span>green box: logs it · red box: a log call omits it · dashed: entered / stored and read later</span></div>
<div id="graph">rendering…</div>
{body}
<script src="https://cdn.jsdelivr.net/npm/@viz-js/viz@3.4.0/lib/viz-standalone.js"></script>
<script>
const DOT = {json.dumps(dot)};
Viz.instance().then(viz => {{ const g = document.getElementById("graph"); g.textContent = ""; g.appendChild(viz.renderSVGElement(DOT)); }});
</script>
</body></html>
"""


SINKS = [
    (subprocess.run, "[subprocess.run]"),
    (subprocess.Popen, "[subprocess.Popen]"),
    (os.system, "[os.system]"),
    (builtins.exec, "[exec]"),
    (builtins.eval, "[eval]"),
    (builtins.input, "[input prompt]"),
    (builtins.print, "[print]"),
]


class ScriptedMessages:
    """Scripted Anthropic `messages` API: tool calls, then a timeout (fallback path), then a text answer."""

    REPLIES = [
        [
            ("bash", {"command": "printf '%0300d' 0; echo CANARY_TOOL_$((0+1))  # CANARY_LLM_1"}),  # output > 200 chars
            ("python", {"code": "result = 'CANARY_TOOL_' + str(2)  # CANARY_LLM_2"}),
        ],
        TimeoutError("scripted timeout"),
        "42. CANARY_LLM_3",
    ]

    def __init__(self):
        self.n = 0

    def create(self, **kwargs):
        reply = self.REPLIES[min(self.n, len(self.REPLIES) - 1)]
        self.n += 1
        if isinstance(reply, Exception):
            raise reply
        if isinstance(reply, str):
            return NS(
                content=[NS(type="text", text=reply)], stop_reason="end_turn", usage=NS(input_tokens=0, output_tokens=0)
            )
        content = [NS(type="tool_use", id=f"toolu_{self.n}_{i}", name=n, input=a) for i, (n, a) in enumerate(reply)]
        return NS(content=content, stop_reason="tool_use", usage=NS(input_tokens=0, output_tokens=0))


def run_toy(hidden: bool) -> Tracer:
    root = Path(__file__).resolve().parent / "toy_agent"
    sys.path.insert(0, str(root))
    os.environ.setdefault("ANTHROPIC_API_KEY", "scripted")
    from agent import ToyAgent

    agent = ToyAgent()
    agent.llm.client = NS(messages=ScriptedMessages())
    if hidden:
        agent.llm.system += f" Working directory: {Path.cwd()}"
    sys.stdin = io.StringIO("no CANARY_USER_1\nCANARY_USER_2\n")
    with Tracer(root, {ScriptedMessages.create.__code__: "[LLM API]"}, SINKS) as tracer:
        print("\nfinal:", agent.run("What is 6 * 7? CANARY_USER_0"))
    return tracer


def main() -> None:
    hidden = "--hidden" in sys.argv
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    scenario = args[0] if args else "toy"
    out = Path(args[1] if len(args) > 1 else "out_flow")
    tracer = {"toy": run_toy}[scenario](hidden)
    rep, dot = tracer.report(), tracer.dot()
    out.mkdir(parents=True, exist_ok=True)
    (out / "flow.dot").write_text(dot, encoding="utf-8")
    (out / "flow.json").write_text(json.dumps(rep, indent=2, ensure_ascii=False), encoding="utf-8")
    (out / "flow.html").write_text(page(f"Data flow: {scenario}", dot, rep), encoding="utf-8")
    print(json.dumps(rep["gaps"], indent=2, ensure_ascii=False))
    print(f"wrote {out / 'flow.html'}")


if __name__ == "__main__":
    main()

"""report.html for wrapper_targets.py: click a node (or a list row) in the call graph -> its CFG and verdict.

Self-contained: the call graph SVG, every CFG SVG and the data are inlined, so the file opens in any browser
offline (double-click it, or `open report.html`). Written by wrapper_targets.py; not meant to be run directly.
"""

import html
import json
import re
from pathlib import Path


def _svg(path: Path):
    if not path.exists():
        return None
    s = path.read_text(encoding="utf-8")
    s = s[s.find("<svg"):]
    s = re.sub(r'<svg([^>]*?)\swidth="[^"]*"', r"<svg\1", s, count=1)
    s = re.sub(r'<svg([^>]*?)\sheight="[^"]*"', r"<svg\1", s, count=1)
    return s


def write(out_dir: Path, title: str, summary: dict, nodes: dict, nid_to_key: dict):
    """nodes: key -> {label, file, line, kind, logged, covered, bypass, sites[], cfg (path rel. to out_dir) | None}"""
    cfgs = {}
    for k, n in nodes.items():
        if n.get("cfg"):
            svg = _svg(out_dir / n["cfg"])
            if svg:
                cfgs[k] = svg
    data = {"summary": summary, "nodes": nodes, "nid": nid_to_key, "cfgs": cfgs}
    graph = _svg(out_dir / "wrapper_targets.svg") or "<p class='empty'>Graphviz가 없어 호출 그래프를 그리지 못했어요.</p>"
    page = (TEMPLATE.replace("__TITLE__", html.escape(title))
            .replace("__GRAPH__", graph)
            .replace("__DATA__", json.dumps(data, ensure_ascii=False).replace("</", "<\\/")))
    (out_dir / "report.html").write_text(page, encoding="utf-8")


TEMPLATE = r"""<!doctype html>
<html lang="ko">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>__TITLE__</title>
<style>
:root {
  --bg: #f6f6f4; --panel: #ffffff; --ink: #1d1d1b; --muted: #6b6b66; --line: #e2e2dd;
  --red: #c4231f; --red-bg: #fdeceb; --green: #1a8a3a; --green-bg: #e7f5ea; --blue: #2f55a8; --blue-bg: #e8eefb;
  --grey-bg: #f0f0ec; --sel: #f2b400; --mono: ui-monospace, "SF Mono", Menlo, Consolas, monospace;
}
@media (prefers-color-scheme: dark) {
  :root { --bg: #161615; --panel: #1f1f1d; --ink: #ececea; --muted: #9a9a94; --line: #34342f;
          --red: #ff7a72; --red-bg: #3a1d1b; --green: #6fd08a; --green-bg: #17301e; --blue: #8fb0ff; --blue-bg: #1c2540;
          --grey-bg: #2a2a27; }
  .canvas svg { filter: invert(0.88) hue-rotate(180deg); }
}
* { box-sizing: border-box; }
html, body { height: 100%; margin: 0; }
body { background: var(--bg); color: var(--ink); font: 14px/1.45 -apple-system, BlinkMacSystemFont, "Apple SD Gothic Neo", "Segoe UI", sans-serif;
       display: grid; grid-template-rows: auto 1fr; }
header { padding: 14px 20px 12px; border-bottom: 1px solid var(--line); background: var(--panel);
         display: flex; flex-wrap: wrap; align-items: baseline; gap: 8px 24px; }
header h1 { font-size: 16px; margin: 0; font-weight: 650; }
.stats { display: flex; flex-wrap: wrap; gap: 6px 18px; color: var(--muted); font-size: 13px; }
.stats b { color: var(--ink); font-variant-numeric: tabular-nums; }
.stats .bad b { color: var(--red); } .stats .good b { color: var(--green); }
main { display: grid; grid-template-columns: 260px minmax(0, 1fr) minmax(0, 1fr); min-height: 0; }
aside, section { min-height: 0; display: flex; flex-direction: column; border-right: 1px solid var(--line); }
section:last-child { border-right: 0; }
.head { padding: 10px 14px; font-size: 12px; text-transform: uppercase; letter-spacing: .06em; color: var(--muted);
        border-bottom: 1px solid var(--line); display: flex; align-items: center; gap: 8px; background: var(--panel); }
.head .sp { flex: 1; }
.head button { font: inherit; text-transform: none; letter-spacing: 0; border: 1px solid var(--line); background: var(--bg);
               color: var(--ink); border-radius: 6px; padding: 2px 8px; cursor: pointer; }
aside .list { overflow: auto; padding: 6px 0 16px; background: var(--panel); flex: 1; }
.group { padding: 12px 14px 4px; font-size: 12px; color: var(--muted); }
.item { display: block; width: 100%; text-align: left; border: 0; background: none; color: inherit; font: inherit;
        padding: 6px 14px; cursor: pointer; border-left: 3px solid transparent; }
.item:hover { background: var(--grey-bg); }
.item.sel { background: var(--grey-bg); border-left-color: var(--sel); }
.item .n { font-family: var(--mono); font-size: 12.5px; word-break: break-all; }
.item .f { color: var(--muted); font-size: 11.5px; font-family: var(--mono); word-break: break-all; }
.pill { display: inline-block; font-size: 11px; padding: 0 6px; border-radius: 9px; margin-left: 4px; vertical-align: 1px; }
.pill.ok { background: var(--green-bg); color: var(--green); } .pill.no { background: var(--red-bg); color: var(--red); }
.pill.log { background: var(--green-bg); color: var(--green); } .pill.tgt { background: var(--red-bg); color: var(--red); }
.canvas { flex: 1; min-height: 0; overflow: hidden; position: relative; background: var(--panel); cursor: grab; touch-action: none; }
.canvas.drag { cursor: grabbing; }
.canvas svg { width: 100%; height: 100%; display: block; }
.canvas .empty, .placeholder { color: var(--muted); padding: 24px; margin: 0; }
#graph g.node { cursor: pointer; }
#graph g.node:hover polygon, #graph g.node:hover ellipse { stroke-width: 3px; }
#graph g.node.sel polygon, #graph g.node.sel ellipse { stroke: var(--sel) !important; stroke-width: 5px; }
#graph g.node.dim { opacity: .35; }
.info { padding: 12px 16px; border-bottom: 1px solid var(--line); background: var(--panel); max-height: 45%; overflow: auto; }
.info h2 { font: 600 14px var(--mono); margin: 0 0 4px; word-break: break-all; }
.info .loc { font: 12px var(--mono); color: var(--muted); margin-bottom: 8px; }
.verdict { display: inline-block; padding: 3px 10px; border-radius: 6px; font-size: 13px; font-weight: 600; margin: 2px 6px 6px 0; }
.verdict.ok { background: var(--green-bg); color: var(--green); } .verdict.no { background: var(--red-bg); color: var(--red); }
.verdict.na { background: var(--grey-bg); color: var(--muted); } .verdict.log { background: var(--green-bg); color: var(--green); }
.verdict.tgt { background: var(--red-bg); color: var(--red); } .verdict.ctx { background: var(--blue-bg); color: var(--blue); }
.info table { border-collapse: collapse; width: 100%; font-size: 12.5px; margin-top: 4px; }
.info td, .info th { text-align: left; padding: 4px 8px 4px 0; border-top: 1px solid var(--line); vertical-align: top; }
.info th { color: var(--muted); font-weight: 500; }
.info td.m { font-family: var(--mono); word-break: break-all; }
.bypass { margin-top: 8px; padding: 8px 10px; background: var(--red-bg); color: var(--red); border-radius: 6px; font: 12px var(--mono); word-break: break-all; }
.hint { font-size: 12px; color: var(--muted); }
@media (max-width: 1100px) {
  main { grid-template-columns: 1fr; grid-auto-rows: minmax(360px, auto); overflow: auto; }
  aside, section { border-right: 0; border-bottom: 1px solid var(--line); }
  aside .list { max-height: 280px; }
  .canvas { min-height: 420px; }
}
</style>
</head>
<body>
<header>
  <h1>__TITLE__</h1>
  <div class="stats" id="stats"></div>
</header>
<main>
  <aside>
    <div class="head">확인 대상</div>
    <div class="list" id="list"></div>
  </aside>
  <section>
    <div class="head">호출 그래프 <span class="sp"></span><span class="hint">스크롤 확대 · 드래그 이동</span><button data-fit="graph">맞춤</button></div>
    <div class="canvas" id="graph">__GRAPH__</div>
  </section>
  <section>
    <div class="info" id="info"><p class="placeholder">왼쪽 목록이나 그래프에서 함수를 누르면 CFG와 판정이 여기에 나와요.</p></div>
    <div class="head">CFG <span class="sp"></span><span class="hint">실제 크기 · 드래그 이동</span><button data-fit="cfg">전체 보기</button></div>
    <div class="canvas" id="cfg"><p class="placeholder">선택한 함수 없음</p></div>
  </section>
</main>
<script>
const D = __DATA__;
const $ = s => document.querySelector(s);
const esc = s => String(s ?? "").replace(/[&<>"]/g, c => ({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;"}[c]));

// ---------- pan / zoom on an <svg> via its viewBox
function zoomable(box, natural) {
  const svg = box.querySelector("svg"); if (!svg) return () => {};
  const vb0 = (svg.getAttribute("viewBox") || "0 0 100 100").split(/[\s,]+/).map(Number);
  let vb = vb0.slice();
  const set = () => svg.setAttribute("viewBox", vb.join(" "));
  const fit = () => { vb = vb0.slice(); set(); };
  if (natural) {  // 1:1 scale, start at the top (CFG entry), centred horizontally if it fits
    const r = box.getBoundingClientRect();
    vb = [vb0[2] > r.width ? vb0[0] : vb0[0] - (r.width - vb0[2]) / 2,
          vb0[3] > r.height ? vb0[1] : vb0[1] - (r.height - vb0[3]) / 2, r.width, r.height];
    set();
  }
  box.addEventListener("wheel", e => {
    e.preventDefault();
    const r = box.getBoundingClientRect(), k = Math.exp(e.deltaY * 0.0015);
    const sx = vb[2] / r.width, sy = vb[3] / r.height, s = Math.max(sx, sy);
    const px = vb[0] + (e.clientX - r.left - (r.width - vb[2] / s) / 2) * s;
    const py = vb[1] + (e.clientY - r.top - (r.height - vb[3] / s) / 2) * s;
    vb = [px - (px - vb[0]) * k, py - (py - vb[1]) * k, vb[2] * k, vb[3] * k]; set();
  }, {passive: false});
  let drag = null, moved = false;
  box.addEventListener("pointerdown", e => { drag = [e.clientX, e.clientY, vb[0], vb[1]]; moved = false; });
  window.addEventListener("pointermove", e => {
    if (!drag) return;
    const dx = e.clientX - drag[0], dy = e.clientY - drag[1];
    if (Math.abs(dx) + Math.abs(dy) > 3) { moved = true; box.classList.add("drag"); }
    const r = box.getBoundingClientRect(), s = Math.max(vb[2] / r.width, vb[3] / r.height);
    vb[0] = drag[2] - dx * s; vb[1] = drag[3] - dy * s; set();
  });
  window.addEventListener("pointerup", () => { drag = null; box.classList.remove("drag"); });
  box.addEventListener("click", e => { if (moved) { e.stopPropagation(); e.preventDefault(); } }, true);
  return fit;
}
const fits = {graph: zoomable($("#graph")), cfg: () => {}};
document.querySelectorAll("[data-fit]").forEach(b => b.onclick = () => fits[b.dataset.fit]());

// ---------- header stats
const S = D.summary;
$("#stats").innerHTML = [
  ["LLM 호출", S.sites], ["wrapper 대상 함수", S.targets],
  S.entry ? ["진입점", S.entry] : null,
  S.entry ? ["로그로 덮임", S.covered, "good"] : null,
  S.entry ? ["미적용", S.not_covered, S.not_covered ? "bad" : "good"] : null,
  ["CFG", Object.keys(D.cfgs).length],
].filter(Boolean).map(([k, v, c]) => `<span class="${c || ""}">${esc(k)} <b>${esc(v)}</b></span>`).join("");

// ---------- list
const keys = Object.keys(D.nodes);
const groups = [
  ["wrapper 대상 (LLM 호출을 직접 가진 함수)", k => D.nodes[k].kind.includes("target")],
  ["이미 있는 wrapper ([LLM_*] 로그)", k => D.nodes[k].logged && !D.nodes[k].kind.includes("target")],
  ["경로 (진입점 → 대상)", k => !D.nodes[k].logged && !D.nodes[k].kind.includes("target")],
];
$("#list").innerHTML = groups.map(([title, f]) => {
  const ks = keys.filter(f).sort((a, b) => {
    const x = D.nodes[a], y = D.nodes[b];
    return (x.covered === false ? 0 : 1) - (y.covered === false ? 0 : 1) || a.localeCompare(b);
  });
  if (!ks.length) return "";
  return `<div class="group">${esc(title)} · ${ks.length}</div>` + ks.map(k => {
    const n = D.nodes[k];
    const pill = n.covered === false ? '<span class="pill no">미적용</span>'
               : n.covered === true ? '<span class="pill ok">덮임</span>' : "";
    return `<button class="item" data-k="${esc(k)}"><div class="n">${esc(n.label)}${pill}</div>
            <div class="f">${esc(n.file)}:${n.line}</div></button>`;
  }).join("");
}).join("");
$("#list").addEventListener("click", e => { const b = e.target.closest(".item"); if (b) select(b.dataset.k, true); });

// ---------- graph clicks
const gnodes = {};
document.querySelectorAll("#graph g.node").forEach(g => {
  const nid = g.querySelector("title")?.textContent.trim();
  const k = D.nid[nid]; if (!k) return;
  (gnodes[k] = gnodes[k] || []).push(g);
  g.addEventListener("click", () => select(k, false));
});

// ---------- select
function select(k, fromList) {
  const n = D.nodes[k]; if (!n) return;
  document.querySelectorAll(".item.sel, #graph g.node.sel").forEach(x => x.classList.remove("sel"));
  document.querySelectorAll(`.item[data-k="${CSS.escape(k)}"]`).forEach(x => {
    x.classList.add("sel"); if (!fromList) x.scrollIntoView({block: "nearest"});
  });
  (gnodes[k] || []).forEach(g => g.classList.add("sel"));

  const badges = [];
  if (n.kind.includes("target")) badges.push('<span class="verdict tgt">wrapper 대상</span>');
  if (n.logged) badges.push('<span class="verdict log">[LLM_*] 로그 있음</span>');
  if (!n.kind.includes("target") && !n.logged) badges.push('<span class="verdict ctx">경로 위 함수</span>');
  if (n.kind.includes("target")) badges.push(
    n.covered === true ? '<span class="verdict ok">모든 경로가 로그를 지남</span>'
    : n.covered === false ? '<span class="verdict no">로그를 거치지 않는 경로 있음</span>'
    : '<span class="verdict na">진입점에서 닿지 않음 (판정 없음)</span>');
  const rows = n.sites.map(s => `<tr><td>L${s.line}</td><td>${esc(s.rule)}</td><td class="m">${esc(s.pattern)}</td><td class="m">${esc(s.detail)}</td></tr>`).join("");
  $("#info").innerHTML = `<h2>${esc(n.label)}</h2><div class="loc">${esc(n.file)}:${n.line}</div>${badges.join("")}
    ${rows ? `<table><tr><th>행</th><th>규칙</th><th>패턴</th><th>호출</th></tr>${rows}</table>` : ""}
    ${n.bypass ? `<div class="bypass">로그 없이 닿는 경로: ${esc(n.bypass)}</div>` : ""}`;

  const box = $("#cfg");
  if (D.cfgs[k]) { box.innerHTML = D.cfgs[k]; fits.cfg = zoomable(box, true); }
  else box.innerHTML = `<p class="placeholder">${
      typeof n.cfg === "string" && n.cfg.startsWith("CFG failed") ? `CFG를 그리지 못했어요 (${esc(n.cfg)}). 소스에서 직접 확인하세요.`
      : n.cfg === false ? "모듈 수준 코드라 CFG가 없어요."
      : n.kind.includes("target") || n.logged ? "CFG가 없어요. --cfg 옵션으로 다시 실행하세요."
      : "경로 표시용 함수라 CFG를 그리지 않았어요."}</p>`;
}
const first = keys.find(k => D.nodes[k].covered === false) || keys.find(k => D.nodes[k].kind.includes("target"));
if (first) select(first, false);
</script>
</body>
</html>
"""

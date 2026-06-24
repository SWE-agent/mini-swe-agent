"""Trajectory visualization (Phase 5 v1): Mermaid, ASCII charts, HTML reports."""

from __future__ import annotations

import html
import webbrowser
from pathlib import Path

from repopilot.eval.loader import RunRecord
from repopilot.trace.parse import extract_pytest_log


def _mermaid_label(text: str, *, limit: int = 48) -> str:
    cleaned = " ".join(text.split())
    if len(cleaned) > limit:
        cleaned = cleaned[: limit - 1] + "…"
    return cleaned.replace('"', "'")


def render_mermaid_source(record: RunRecord) -> str:
    """Return Mermaid flowchart source (no fences)."""
    if not record.steps:
        return "flowchart LR\n    empty[No steps recorded]"

    failed = record.failed_step
    lines = ["flowchart LR"]
    node_ids: list[str] = []

    for step in record.steps:
        step_num = step.get("step")
        stage = step.get("stage", "other")
        node_id = f"s{step_num}"
        node_ids.append(node_id)
        label = _mermaid_label(f"Step {step_num}<br/>{stage}")
        lines.append(f'    {node_id}["{label}"]')

    for left, right in zip(node_ids, node_ids[1:]):
        lines.append(f"    {left} --> {right}")

    if failed is not None:
        lines.append("    classDef failed fill:#fde8e8,stroke:#c0392b,stroke-width:2px")
        lines.append(f"    class s{failed} failed")

    return "\n".join(lines)


def render_mermaid_timeline(record: RunRecord) -> str:
    """Render a fenced Mermaid block for markdown reports."""
    return f"```mermaid\n{render_mermaid_source(record)}\n```\n"


def render_ascii_bar_chart(counts: dict[str, int], *, width: int = 32, title: str | None = None) -> str:
    """Render a horizontal ASCII bar chart for count dictionaries."""
    if not counts:
        body = "(none)"
    else:
        max_val = max(counts.values())
        label_width = max(len(label) for label in counts)
        rows: list[str] = []
        for label, count in counts.items():
            bar_len = round(width * count / max_val) if max_val else 0
            rows.append(f"{label:<{label_width}} | {'█' * bar_len} {count}")
        body = "\n".join(rows)

    if title:
        return f"{title}\n\n{body}\n"
    return body + "\n"


def render_failure_distribution_charts(breakdown: dict) -> str:
    """Combine ASCII charts for failure category and stage."""
    sections: list[str] = ["## Failure distribution", ""]
    if breakdown.get("by_category"):
        sections.append("### By category")
        sections.append("")
        sections.append("```")
        sections.append(render_ascii_bar_chart(breakdown["by_category"]).rstrip())
        sections.append("```")
        sections.append("")
    if breakdown.get("by_stage"):
        sections.append("### By stage")
        sections.append("")
        sections.append("```")
        sections.append(render_ascii_bar_chart(breakdown["by_stage"]).rstrip())
        sections.append("```")
        sections.append("")
    if len(sections) <= 2:
        return ""
    return "\n".join(sections)


def _read_optional(path: Path) -> str:
    return path.read_text() if path.is_file() else ""


def _pytest_panel_html(record: RunRecord) -> str:
    blocks: list[str] = []
    for run in record.pytest_runs:
        phase = html.escape(str(run.get("phase", "unknown")))
        rc = run.get("returncode")
        status = "pass" if rc == 0 else "fail" if rc is not None else "unknown"
        summary = html.escape(run.get("summary") or "")
        log = run.get("log") or ""
        if log:
            log = extract_pytest_log(log) if "test session starts" in log.lower() else log
        log_text = html.escape(log[:4000] + ("…" if len(log) > 4000 else ""))
        blocks.append(
            f'<section class="panel pytest {status}">'
            f"<h3>{phase} — exit {rc if rc is not None else '?'} {summary}</h3>"
            f'<pre class="log">{log_text}</pre></section>'
        )
    verify_log = _read_optional(record.run_dir / "verify_test.log")
    if verify_log.strip():
        blocks.append(
            '<section class="panel pytest verify">'
            "<h3>Runner verify</h3>"
            f'<pre class="log">{html.escape(verify_log[:4000])}</pre></section>'
        )
    return "\n".join(blocks)


def _steps_html(record: RunRecord) -> str:
    blocks: list[str] = []
    failed = record.failed_step
    for step in record.steps:
        step_num = step.get("step")
        stage = html.escape(str(step.get("stage", "other")))
        css = "step failed" if step_num == failed else "step"
        reasoning = html.escape(step.get("reasoning") or "")
        files = ", ".join(html.escape(p) for p in (step.get("files_touched") or []))
        tools: list[str] = []
        for tc in step.get("tool_calls", []):
            cmd = html.escape((tc.get("command") or "").replace("\n", " ")[:160])
            rc = tc.get("returncode")
            tools.append(f'<li><code>[{rc}]</code> {cmd}</li>')
        tool_html = f"<ul>{''.join(tools)}</ul>" if tools else ""
        files_html = f"<p><strong>Files:</strong> {files}</p>" if files else ""
        blocks.append(
            f'<article class="{css}">'
            f"<h3>Step {step_num} <span class=\"stage\">{stage}</span></h3>"
            f"<p>{reasoning}</p>{files_html}{tool_html}</article>"
        )
    return "\n".join(blocks)


def render_run_html(record: RunRecord) -> str:
    """Self-contained HTML report for a single run."""
    from repopilot.eval.trajectory_analysis import trajectory_metrics

    metrics = trajectory_metrics(record)
    mermaid = render_mermaid_source(record)
    patch = _read_optional(record.run_dir / "patch.diff")
    if patch.startswith("# No patch"):
        patch = ""

    outcome_class = "success" if record.outcome == "success" else "failure"
    meta_rows = [
        ("Task", record.task_id),
        ("Agent mode", record.agent_mode),
        ("Model", record.model),
        ("Outcome", record.outcome),
        ("Verify passed", record.tests_passed),
        ("Steps", metrics["step_count"]),
        ("Cost", f"${record.instance_cost:.4f}"),
        ("Steps to first edit", metrics["steps_to_first_edit"] or "—"),
        ("Files touched", metrics["files_touched_count"]),
    ]
    if record.failure_category:
        meta_rows.extend(
            [
                ("Failure category", record.failure_category),
                ("Failure stage", record.failure_stage or "—"),
                ("Failed step", record.failed_step or "—"),
                ("Failure message", record.failure_message or "—"),
            ]
        )

    meta_html = "".join(
        f"<tr><th>{html.escape(str(k))}</th><td>{html.escape(str(v))}</td></tr>" for k, v in meta_rows
    )

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{html.escape(record.task_id)} — RepoPilot Run View</title>
  <style>
    :root {{
      --bg: #0f1419; --surface: #1a2332; --text: #e6edf3; --muted: #8b949e;
      --accent: #58a6ff; --pass: #3fb950; --fail: #f85149; --border: #30363d;
    }}
    * {{ box-sizing: border-box; }}
    body {{ font-family: ui-sans-serif, system-ui, sans-serif; background: var(--bg); color: var(--text); margin: 0; line-height: 1.5; }}
    header {{ background: var(--surface); border-bottom: 1px solid var(--border); padding: 1.25rem 2rem; }}
    header h1 {{ margin: 0 0 .25rem; font-size: 1.35rem; }}
    .badge {{ display: inline-block; padding: .15rem .55rem; border-radius: 999px; font-size: .8rem; font-weight: 600; }}
    .badge.success {{ background: #238636; color: #fff; }}
    .badge.failure {{ background: #da3633; color: #fff; }}
    main {{ max-width: 1100px; margin: 0 auto; padding: 1.5rem 2rem 3rem; }}
    section {{ margin-bottom: 2rem; }}
    h2 {{ font-size: 1.1rem; border-bottom: 1px solid var(--border); padding-bottom: .4rem; }}
    table.meta {{ width: 100%; border-collapse: collapse; font-size: .92rem; }}
    table.meta th {{ text-align: left; color: var(--muted); width: 11rem; padding: .35rem .5rem .35rem 0; vertical-align: top; }}
    table.meta td {{ padding: .35rem 0; }}
    .mermaid-wrap {{ background: var(--surface); border: 1px solid var(--border); border-radius: 8px; padding: 1rem; overflow-x: auto; }}
    article.step {{ background: var(--surface); border: 1px solid var(--border); border-radius: 8px; padding: 1rem; margin-bottom: .75rem; }}
    article.step.failed {{ border-color: var(--fail); box-shadow: inset 3px 0 0 var(--fail); }}
    .stage {{ color: var(--accent); font-weight: normal; font-size: .85rem; }}
    .panel {{ background: var(--surface); border: 1px solid var(--border); border-radius: 8px; padding: 1rem; margin-bottom: .75rem; }}
    .panel.pass {{ border-left: 3px solid var(--pass); }}
    .panel.fail {{ border-left: 3px solid var(--fail); }}
    pre.log, pre.patch {{ background: #010409; border: 1px solid var(--border); border-radius: 6px; padding: .75rem; overflow-x: auto; font-size: .78rem; white-space: pre-wrap; word-break: break-word; }}
    ul {{ margin: .5rem 0 0; padding-left: 1.2rem; }}
    code {{ font-size: .85em; }}
  </style>
</head>
<body>
  <header>
    <h1>{html.escape(record.task_id)}</h1>
    <span class="badge {outcome_class}">{html.escape(record.outcome)}</span>
    <span style="color:var(--muted); margin-left:.5rem">{html.escape(record.agent_mode)} · {html.escape(record.model)}</span>
  </header>
  <main>
    <section>
      <h2>Summary</h2>
      <table class="meta">{meta_html}</table>
    </section>
    <section>
      <h2>Trajectory timeline</h2>
      <div class="mermaid-wrap"><pre class="mermaid">{mermaid}</pre></div>
    </section>
    <section>
      <h2>Agent steps</h2>
      {_steps_html(record)}
    </section>
    <section>
      <h2>Pytest</h2>
      {_pytest_panel_html(record) or '<p class="muted">No pytest output recorded.</p>'}
    </section>
    <section>
      <h2>Patch</h2>
      <pre class="patch">{html.escape(patch or '(no patch extracted)')}</pre>
    </section>
  </main>
  <script type="module">
    import mermaid from 'https://cdn.jsdelivr.net/npm/mermaid@11/dist/mermaid.esm.min.mjs';
    mermaid.initialize({{ startOnLoad: true, theme: 'dark', securityLevel: 'strict' }});
  </script>
</body>
</html>
"""


def write_run_view(record: RunRecord, output_path: Path | None = None) -> Path:
    """Write HTML view for a run; default runs/eval/{task_id}/view.html."""
    if output_path is None:
        runs_root = record.run_dir.parent if record.run_dir.parent.name != "history" else record.run_dir.parent.parent.parent
        output_path = runs_root / "eval" / record.task_id / "view.html"
    output_path = output_path.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(render_run_html(record))
    return output_path


def open_run_view(path: Path) -> None:
    webbrowser.open(path.as_uri())

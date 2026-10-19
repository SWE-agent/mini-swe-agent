# /// script
# requires-python = ">=3.10"
# dependencies = []
# ///
"""Agent Observatory dashboard: shows the LLM calls and tool executions an agent reports through observatory_log.py.

    uv run observatory/dashboard.py [--runs observatory/runs] [--port 8000]

It knows nothing about any agent. It only reads the run files (JSON lines) that observatory_log.py appends to the
directory named by OBSERVATORY_RUNS, so start the agent as usual with that variable set, e.g.
`OBSERVATORY_RUNS=$PWD/observatory/runs <your agent command>`, and watch the history appear live.
The left panel shows observatory/diagram/agent-architecture.html when it exists.
"""

import argparse
import json
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

KIT = Path(__file__).resolve().parent
LIVE_SECONDS = 10
NO_DIAGRAM = b"<!doctype html><meta charset=utf-8><body style='font:14px system-ui;color:#8a9ab3;background:#020617;padding:24px'>No architecture diagram yet. Run observatory/run.py with Archify available to generate it.</body>"


class Stream:
    """Turns the complete lines of a run file into numbered events; call poll() again to get what was appended since."""

    def __init__(self, path: Path) -> None:
        self.path, self.offset, self.seq, self.step, self.query_at = path, 0, 0, None, None

    def poll(self) -> list[dict]:
        with self.path.open("rb") as file:
            file.seek(self.offset)
            chunk = file.read()
        complete = chunk[: chunk.rfind(b"\n") + 1]
        self.offset += len(complete)
        events = []
        for line in complete.splitlines():
            try:
                events.append(self.normalize(json.loads(line)))
            except (json.JSONDecodeError, KeyError):
                continue  # the file is written by other processes: skip a line that is not an event
        return events

    def normalize(self, raw: dict) -> dict:
        kind, step, data = raw["kind"], raw.get("step"), raw.get("data") or {}
        if kind == "llm_query":
            self.step, self.query_at = step, raw["t"]
        elif step is None:
            step = self.step
        if kind in ("llm_response", "llm_error") and self.query_at is not None and "duration_seconds" not in data:
            data = {**data, "duration_seconds": round(raw["t"] - self.query_at, 3)}
        self.seq += 1
        return {"seq": self.seq - 1, "t": raw["t"], "kind": kind, "step": step, "data": data}


def list_runs(directory: Path) -> list[dict]:
    files = (
        sorted(directory.glob("*.jsonl"), key=lambda p: p.stat().st_mtime, reverse=True) if directory.exists() else []
    )
    return [
        {
            "id": p.stem,
            "events": len(p.read_bytes().splitlines()),
            "modified": p.stat().st_mtime,
            "live": time.time() - p.stat().st_mtime < LIVE_SECONDS,
        }
        for p in files
    ]


class Handler(BaseHTTPRequestHandler):
    runs_dir = KIT / "runs"

    def log_message(self, *args) -> None:
        pass

    def reply(self, status: int, content_type: str, body: bytes, headers: dict | None = None) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-cache")
        for name, value in (headers or {}).items():
            self.send_header(name, value)
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        url = urlparse(self.path)
        run = (parse_qs(url.query).get("run") or ["latest"])[0]
        if url.path == "/":
            self.reply(200, "text/html; charset=utf-8", (KIT / "static/index.html").read_bytes())
        elif url.path == "/architecture":
            diagram = KIT / "diagram/agent-architecture.html"
            self.reply(200, "text/html; charset=utf-8", diagram.read_bytes() if diagram.exists() else NO_DIAGRAM)
        elif url.path == "/api/runs":
            self.reply(200, "application/json", json.dumps(list_runs(self.runs_dir)).encode())
        elif url.path in ("/api/events", "/api/trace"):
            runs = list_runs(self.runs_dir)
            match = next((r for r in runs if run == "latest" or r["id"] == run), None)
            if match is None:
                self.reply(404, "application/json", b'{"detail": "No such run"}')
            elif url.path == "/api/trace":
                body = json.dumps(
                    {"run": match["id"], "events": Stream(self.runs_dir / f"{match['id']}.jsonl").poll()},
                    ensure_ascii=False,
                    indent=2,
                )
                self.reply(
                    200,
                    "application/json",
                    body.encode(),
                    {"Content-Disposition": f'attachment; filename="trace-{match["id"]}.json"'},
                )
            else:
                self.stream(self.runs_dir / f"{match['id']}.jsonl")
        else:
            self.reply(404, "text/plain", b"Not found")

    def stream(self, path: Path) -> None:
        """Server-sent events: replay the run so far, then send new events as they are appended."""
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        stream, quiet = Stream(path), 0
        try:
            while True:
                events = stream.poll()
                quiet = 0 if events else quiet + 1
                for event in events:
                    self.wfile.write(f"data: {json.dumps(event, ensure_ascii=False)}\n\n".encode())
                if events or quiet % 50 == 0:
                    if not events:
                        self.wfile.write(b": keepalive\n\n")
                    self.wfile.flush()
                time.sleep(0.3)
        except (BrokenPipeError, ConnectionResetError):
            pass  # the browser tab closed


def make_server(runs_dir: Path, port: int) -> ThreadingHTTPServer:
    handler = type("RunsHandler", (Handler,), {"runs_dir": runs_dir})
    server = ThreadingHTTPServer(("127.0.0.1", port), handler)
    server.daemon_threads = True
    return server


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--runs", type=Path, default=KIT / "runs", help="directory of run files (default: observatory/runs)"
    )
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    try:
        server = make_server(args.runs.resolve(), args.port)
    except OSError as error:
        raise SystemExit(f"Cannot listen on port {args.port} ({error}): pick a free one with --port.")
    print(f"Dashboard: http://127.0.0.1:{args.port}  (Ctrl-C to stop)")
    print(f"Run your agent with:  OBSERVATORY_RUNS={args.runs.resolve()} <your agent command>")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass

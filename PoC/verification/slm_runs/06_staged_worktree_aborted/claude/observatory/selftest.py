# /// script
# requires-python = ">=3.10"
# dependencies = []
# ///
"""Self-test of the kit's logger and dashboard (no agent, no network): uv run observatory/selftest.py"""

import json
import os
import shutil
import subprocess
import sys
import tempfile
import threading
import urllib.request
from pathlib import Path

KIT = Path(__file__).resolve().parent
sys.path.insert(0, str(KIT))
import dashboard  # noqa: E402
import observatory_log as log  # noqa: E402


def read(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines()]


def test_logger_is_silent_without_the_variable(tmp: Path) -> None:
    os.environ.pop("OBSERVATORY_RUNS", None)
    log.llm_query(1, model="m", messages=2)
    assert not list(tmp.iterdir())


def test_logger_writes_one_line_per_event(tmp: Path) -> None:
    os.environ["OBSERVATORY_RUNS"] = str(tmp / "runs")
    log._files.clear()
    log.llm_query(1, model="m", messages=2, purpose="action")
    log.llm_response(1, actions=[{"bash": {"command": "ls"}}], cost=0.01)
    start = log.tool_start("bash", {"command": "ls"})
    log.tool_end(start, "bash", result="exit 0")
    log.tool_error(log.tool_start("bash", {"command": "x"}), "bash", RuntimeError("boom"))
    log.llm_error(2, ValueError("bad"))
    (file,) = (tmp / "runs").glob("*.jsonl")
    events = read(file)
    assert [(e["kind"], e["step"]) for e in events] == [
        ("llm_query", 1),
        ("llm_response", 1),
        ("tool_start", None),
        ("tool_end", None),
        ("tool_start", None),
        ("tool_error", None),
        ("llm_error", 2),
    ]
    assert events[0]["data"] == {"model": "m", "messages": 2, "purpose": "action"}
    assert {e["at"].rsplit(":", 1)[0].rsplit("/", 1)[-1] for e in events} == {"selftest.py"}
    assert events[0]["at"] != events[1]["at"]
    assert events[1]["data"]["actions"] == [{"bash": {"command": "ls"}}] and events[1]["data"]["cost"] == 0.01
    assert events[3]["data"]["result"] == "exit 0" and events[3]["data"]["duration_seconds"] >= 0
    assert events[5]["data"]["error_type"] == "RuntimeError" and events[5]["data"]["error"] == "boom"


def test_logger_scrubs_secrets_and_truncates(tmp: Path) -> None:
    os.environ["OBSERVATORY_RUNS"] = str(tmp / "scrub")
    os.environ["MY_API_KEY"] = "sk-secret-value-123"
    log._files.clear()
    log.tool_start("bash", {"command": "curl -H sk-secret-value-123 " + "x" * 5000})
    (file,) = (tmp / "scrub").glob("*.jsonl")
    line = file.read_text()
    assert "sk-secret-value-123" not in line and "<redacted>" in line
    assert len(json.loads(line)["data"]["params"]["command"]) <= log.MAX_CHARS + 1
    del os.environ["MY_API_KEY"]


def test_stream_numbers_events_and_assigns_steps(tmp: Path) -> None:
    path = tmp / "s.jsonl"
    lines = [
        {"t": 10.0, "kind": "llm_query", "step": 3, "data": {}},
        {"t": 10.5, "kind": "tool_start", "step": None, "data": {"action": "a"}},
        {"t": 11.0, "kind": "llm_response", "step": 3, "data": {"actions": []}},
    ]
    path.write_text("\n".join(json.dumps(x) for x in lines) + "\n{not json}\n" + '{"t": 12, "kind": "llm_qu')
    stream = dashboard.Stream(path)
    events = stream.poll()
    assert [(e["seq"], e["kind"], e["step"]) for e in events] == [
        (0, "llm_query", 3),
        (1, "tool_start", 3),
        (2, "llm_response", 3),
    ]
    assert events[2]["data"]["duration_seconds"] == 1.0
    with path.open("a") as file:
        file.write('ery", "step": 4, "data": {}}\n')
    assert [e["step"] for e in stream.poll()] == [4]


def test_server(tmp: Path) -> None:
    runs = tmp / "srv"
    runs.mkdir()
    (runs / "20260101-000000-1.jsonl").write_text(
        json.dumps({"t": 1.0, "kind": "llm_query", "step": 1, "data": {"model": "m"}}) + "\n"
    )
    server = dashboard.make_server(runs, 0)
    port = server.server_address[1]
    threading.Thread(target=server.serve_forever, daemon=True).start()
    get = lambda path: urllib.request.urlopen(f"http://127.0.0.1:{port}{path}", timeout=5)
    try:
        assert b"Agent Observatory" in get("/").read()
        assert b"No architecture diagram" in get("/architecture").read() or b"<svg" in get("/architecture").read()
        assert json.load(get("/api/runs"))[0]["id"] == "20260101-000000-1"
        with get("/api/events?run=latest") as stream:
            assert json.loads(stream.readline().split(b"data: ")[1])["kind"] == "llm_query"
        assert json.load(get("/api/trace"))["events"][0]["data"] == {"model": "m"}
        try:
            get("/api/events?run=missing")
            raise AssertionError("expected 404")
        except urllib.error.HTTPError as error:
            assert error.code == 404
    finally:
        server.shutdown()


def test_page_script_is_valid_javascript(tmp: Path) -> None:
    node = shutil.which("node")
    if node is None:
        print("  (node not found: skipping the JavaScript syntax check)")
        return
    html = (KIT / "static/index.html").read_text()
    script = html[html.index("<script>") + 8 : html.index("</script>")]
    (tmp / "page.js").write_text(script)
    check = subprocess.run([node, "--check", str(tmp / "page.js")], capture_output=True, text=True)
    assert check.returncode == 0, check.stderr


if __name__ == "__main__":
    tests = [t for name, t in sorted(globals().items()) if name.startswith("test_")]
    for test in tests:
        with tempfile.TemporaryDirectory() as tmp:
            test(Path(tmp))
        print(f"ok  {test.__name__}")
    print(f"\n{len(tests)} checks passed.")

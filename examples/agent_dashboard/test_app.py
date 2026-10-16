from app import RunRequest, TraceHub, execute_agent

from minisweagent.models.test_models import DeterministicModel, make_output

SUBMIT = "echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT\necho done"


def run(tmp_path, outputs: list[dict], max_steps: int = 5) -> TraceHub:
    hub = TraceHub(tmp_path)
    request = RunRequest(task="Say hi", max_steps=max_steps)
    hub.start(request)
    execute_agent(hub, request, DeterministicModel(outputs=outputs))
    return hub


def test_run_is_traced_from_boundary_logs(tmp_path):
    hub = run(tmp_path, [make_output("look", [{"command": "echo hi"}]), make_output("submit", [{"command": SUBMIT}])])
    events = hub.run.events
    assert [(e.kind, e.step) for e in events if e.kind != "step"] == [
        ("llm_query", 1),
        ("llm_response", 1),
        ("tool_start", 1),
        ("tool_end", 1),
        ("llm_query", 2),
        ("llm_response", 2),
        ("tool_start", 2),
        ("tool_error", 2),
        ("run", None),
    ]
    assert [e.data["type"] for e in events if e.kind == "step"] == ["thought", "observation", "thought", "exit"]
    assert next(e for e in events if e.kind == "tool_end").data.keys() == {"action", "returncode", "duration_seconds"}
    assert [
        (e.data["action"], e.data.get("returncode", e.data.get("error_type")))
        for e in events
        if e.kind in ("tool_end", "tool_error")
    ] == [("bash", 0), ("bash", "Submitted")]
    assert "hi" in next(e for e in events if e.kind == "step" and e.data["type"] == "observation").data["content"]
    assert events[-1].data | {"cost": 0} == {
        "exit_status": "Submitted",
        "success": True,
        "final_result": "done\n",
        "cost": 0,
    }
    assert (tmp_path / f"{hub.run.id}.json").exists() and (tmp_path / f"{hub.run.id}-workspace").is_dir()


def test_step_limit_ends_the_run_as_incomplete(tmp_path):
    hub = run(tmp_path, [make_output("look", [{"command": "echo hi"}])], max_steps=1)
    assert [e.kind for e in hub.run.events].count("llm_query") == 1
    assert hub.run.events[-2].data["exit_status"] == "LimitsExceeded"
    assert (hub.run.events[-1].data["exit_status"], hub.run.events[-1].data["success"]) == ("LimitsExceeded", None)

import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from minisweagent.models.openrouter_model import OpenRouterAPIError, OpenRouterContextWindowError, OpenRouterModel
from minisweagent.models.openrouter_response_model import OpenRouterResponseModel
from minisweagent.models.openrouter_textbased_model import OpenRouterTextbasedModel
from minisweagent.models.requesty_model import RequestyAPIError, RequestyContextWindowError, RequestyModel


@pytest.fixture
def error_server():
    state = {"requests": 0, "status": 400, "payload": {}}

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            state["requests"] += 1
            body = json.dumps(state["payload"]).encode()
            self.send_response(state["status"])
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield server, state
    server.shutdown()
    server.server_close()
    thread.join()


@pytest.mark.parametrize(
    ("model_class", "error_class", "payload"),
    [
        pytest.param(
            OpenRouterModel,
            OpenRouterContextWindowError,
            {"error": {"message": "provider error", "metadata": {"error_type": "context_length_exceeded"}}},
            id="openrouter-metadata-error-type",
        ),
        pytest.param(
            OpenRouterTextbasedModel,
            OpenRouterContextWindowError,
            {"error": {"message": "provider error", "type": "context_length_exceeded"}},
            id="openrouter-error-type",
        ),
        pytest.param(
            OpenRouterResponseModel,
            OpenRouterContextWindowError,
            {"error_type": "context_length_exceeded", "error": {"code": "invalid_prompt"}},
            id="openrouter-responses-error-type",
        ),
        pytest.param(
            RequestyModel,
            RequestyContextWindowError,
            {"error": {"message": "provider error", "code": "context_length_exceeded"}},
            id="requesty-error-code",
        ),
        pytest.param(
            RequestyModel,
            RequestyContextWindowError,
            {"error": {"message": "This model has a maximum context length of 128000 tokens"}},
            id="requesty-message-fallback",
        ),
    ],
)
def test_context_window_errors_abort_without_retry(error_server, monkeypatch, model_class, error_class, payload):
    server, state = error_server
    state["payload"] = payload
    monkeypatch.setenv("MSWEA_MODEL_RETRY_STOP_AFTER_ATTEMPT", "3")
    monkeypatch.setattr(time, "sleep", lambda _: None)
    model = model_class(model_name="test")
    model._api_url = f"http://127.0.0.1:{server.server_port}/v1/chat/completions"

    with pytest.raises(error_class, match="HTTP 400") as exc_info:
        model.query([{"role": "user", "content": "oversized"}])

    assert type(exc_info.value) is error_class
    assert state["requests"] == 1


@pytest.mark.parametrize(
    ("model_class", "error_class", "status", "payload"),
    [
        pytest.param(
            OpenRouterModel,
            OpenRouterAPIError,
            400,
            {"error": {"code": "invalid_request", "message": "model is missing"}},
            id="openrouter-generic-400",
        ),
        pytest.param(
            OpenRouterModel,
            OpenRouterAPIError,
            500,
            {"error": {"message": "server error"}},
            id="openrouter-500",
        ),
        pytest.param(
            RequestyModel,
            RequestyAPIError,
            400,
            {"error": {"code": "token_limit_exceeded", "message": "token budget exceeded"}},
            id="requesty-unrelated-400",
        ),
        pytest.param(
            RequestyModel,
            RequestyAPIError,
            500,
            {"error": {"message": "server error"}},
            id="requesty-500",
        ),
    ],
)
def test_other_api_errors_remain_retryable(error_server, monkeypatch, model_class, error_class, status, payload):
    server, state = error_server
    state.update({"status": status, "payload": payload})
    monkeypatch.setenv("MSWEA_MODEL_RETRY_STOP_AFTER_ATTEMPT", "2")
    monkeypatch.setattr(time, "sleep", lambda _: None)
    model = model_class(model_name="test")
    model._api_url = f"http://127.0.0.1:{server.server_port}/v1/chat/completions"

    with pytest.raises(error_class) as exc_info:
        model.query([{"role": "user", "content": "test"}])

    assert type(exc_info.value) is error_class
    assert state["requests"] == 2

import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from unittest.mock import MagicMock, patch

import litellm
import openai
import pytest

from minisweagent.exceptions import FormatError, ProviderTimeout
from minisweagent.models.litellm_model import LitellmModel, LitellmModelConfig
from minisweagent.models.utils.actions_toolcall import BASH_TOOL


class TestLitellmModelConfig:
    def test_default_format_error_template(self):
        assert LitellmModelConfig(model_name="test").format_error_template == "{{ error }}"

    def test_default_provider_timeout(self):
        assert LitellmModelConfig(model_name="test").provider_timeout == 5.0


def _mock_litellm_response(tool_calls):
    mock_response = MagicMock()
    mock_response.choices = [MagicMock()]
    mock_response.choices[0].message.tool_calls = tool_calls
    mock_response.choices[0].message.model_dump.return_value = {"role": "assistant", "content": None}
    mock_response.model_dump.return_value = {}
    return mock_response


class TestLitellmModel:
    @patch("minisweagent.models.litellm_model.litellm.completion")
    @patch("minisweagent.models.litellm_model.litellm.cost_calculator.completion_cost")
    def test_query_includes_bash_tool(self, mock_cost, mock_completion):
        tool_call = MagicMock()
        tool_call.function.name = "bash"
        tool_call.function.arguments = '{"command": "echo test"}'
        tool_call.id = "call_1"
        mock_completion.return_value = _mock_litellm_response([tool_call])
        mock_cost.return_value = 0.001

        model = LitellmModel(model_name="gpt-4")
        model.query([{"role": "user", "content": "test"}])

        mock_completion.assert_called_once()
        assert mock_completion.call_args.kwargs["tools"] == [BASH_TOOL]
        assert "timeout" not in mock_completion.call_args.kwargs

    @patch("minisweagent.models.litellm_model.litellm.completion")
    @patch("minisweagent.models.litellm_model.litellm.cost_calculator.completion_cost")
    def test_query_includes_stream_provider_timeout(self, mock_cost, mock_completion):
        tool_call = MagicMock()
        tool_call.function.name = "bash"
        tool_call.function.arguments = '{"command": "echo test"}'
        tool_call.id = "call_1"
        mock_completion.return_value = _mock_litellm_response([tool_call])
        mock_cost.return_value = 0.001

        model = LitellmModel(model_name="gpt-4", model_kwargs={"stream": True})
        model.query([{"role": "user", "content": "test"}])

        timeout = mock_completion.call_args.kwargs["timeout"]
        assert isinstance(timeout, openai.Timeout)
        assert timeout.connect == 10.0
        assert timeout.read == 5.0

    @patch("minisweagent.models.litellm_model.litellm.completion")
    @patch("minisweagent.models.litellm_model.litellm.cost_calculator.completion_cost")
    def test_query_preserves_explicit_timeout(self, mock_cost, mock_completion):
        tool_call = MagicMock()
        tool_call.function.name = "bash"
        tool_call.function.arguments = '{"command": "echo test"}'
        tool_call.id = "call_1"
        mock_completion.return_value = _mock_litellm_response([tool_call])
        mock_cost.return_value = 0.001

        model = LitellmModel(model_name="gpt-4", model_kwargs={"timeout": 30})
        model.query([{"role": "user", "content": "test"}])

        assert mock_completion.call_args.kwargs["timeout"] == 30

    @patch("minisweagent.models.litellm_model.litellm.completion")
    @patch("minisweagent.models.litellm_model.litellm.cost_calculator.completion_cost")
    def test_query_preserves_explicit_request_timeout(self, mock_cost, mock_completion):
        tool_call = MagicMock()
        tool_call.function.name = "bash"
        tool_call.function.arguments = '{"command": "echo test"}'
        tool_call.id = "call_1"
        mock_completion.return_value = _mock_litellm_response([tool_call])
        mock_cost.return_value = 0.001

        model = LitellmModel(model_name="gpt-4", model_kwargs={"request_timeout": 30})
        model.query([{"role": "user", "content": "test"}])

        assert "timeout" not in mock_completion.call_args.kwargs
        assert mock_completion.call_args.kwargs["request_timeout"] == 30

    @patch("minisweagent.models.litellm_model.litellm.completion")
    def test_query_timeout_raises_provider_timeout(self, mock_completion):
        mock_completion.side_effect = TimeoutError("request timed out")

        model = LitellmModel(model_name="gpt-4")
        with pytest.raises(ProviderTimeout) as exc:
            model.query([{"role": "user", "content": "test"}])
        assert exc.value.messages[0]["role"] == "exit"
        assert exc.value.messages[0]["extra"]["exit_status"] == "ProviderTimeout"

    @pytest.mark.parametrize(
        "timeout_error",
        [
            TimeoutError("request timed out"),
            litellm.exceptions.Timeout("request timed out", "gpt-4", "openai"),
        ],
    )
    @patch("minisweagent.models.litellm_model.litellm.completion")
    def test_query_recognizes_provider_timeout_types(self, mock_completion, timeout_error):
        mock_completion.side_effect = timeout_error

        model = LitellmModel(model_name="gpt-4")
        with pytest.raises(ProviderTimeout):
            model.query([{"role": "user", "content": "test"}])

    def test_query_times_out_when_stream_stalls_mid_tool_call(self):
        class StalledToolCallStream(BaseHTTPRequestHandler):
            def do_POST(self):
                self.server.seen_request = True
                self.server.request_body = json.loads(
                    self.rfile.read(int(self.headers.get("content-length", "0"))).decode()
                )
                self.send_response(200)
                self.send_header("content-type", "text/event-stream")
                self.end_headers()
                for event in [
                    {
                        "id": "chatcmpl-test",
                        "object": "chat.completion.chunk",
                        "choices": [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}],
                    },
                    {
                        "id": "chatcmpl-test",
                        "object": "chat.completion.chunk",
                        "choices": [
                            {
                                "index": 0,
                                "delta": {
                                    "tool_calls": [
                                        {
                                            "index": 0,
                                            "id": "call_1",
                                            "type": "function",
                                            "function": {"name": "bash", "arguments": ""},
                                        }
                                    ]
                                },
                                "finish_reason": None,
                            }
                        ],
                    },
                ]:
                    self.wfile.write(f"data: {json.dumps(event)}\n\n".encode())
                    self.wfile.flush()
                time.sleep(5)

            def log_message(self, *args):
                pass

        server = ThreadingHTTPServer(("127.0.0.1", 0), StalledToolCallStream)
        server.seen_request = False
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            model = LitellmModel(
                model_name="openai/gpt-4o",
                provider_timeout=0.2,
                model_kwargs={
                    "api_key": "fake-key",
                    "api_base": f"http://127.0.0.1:{server.server_port}/v1",
                    "stream": True,
                },
                cost_tracking="ignore_errors",
            )
            start = time.monotonic()
            with pytest.raises(ProviderTimeout):
                model.query([{"role": "user", "content": "test"}])
            assert server.seen_request
            assert server.request_body["stream"] is True
            assert time.monotonic() - start < 4
        finally:
            server.shutdown()
            server.server_close()

    def test_query_accumulates_streamed_tool_call_arguments(self):
        class ToolCallStream(BaseHTTPRequestHandler):
            def do_POST(self):
                self.rfile.read(int(self.headers.get("content-length", "0")))
                self.send_response(200)
                self.send_header("content-type", "text/event-stream")
                self.end_headers()
                for event in [
                    {
                        "id": "chatcmpl-test",
                        "object": "chat.completion.chunk",
                        "choices": [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}],
                    },
                    {
                        "id": "chatcmpl-test",
                        "object": "chat.completion.chunk",
                        "choices": [
                            {
                                "index": 0,
                                "delta": {
                                    "tool_calls": [
                                        {
                                            "index": 0,
                                            "id": "call_1",
                                            "type": "function",
                                            "function": {"name": "bash", "arguments": ""},
                                        }
                                    ]
                                },
                                "finish_reason": None,
                            }
                        ],
                    },
                    {
                        "id": "chatcmpl-test",
                        "object": "chat.completion.chunk",
                        "choices": [
                            {
                                "index": 0,
                                "delta": {
                                    "tool_calls": [
                                        {
                                            "index": 0,
                                            "id": "call_1",
                                            "type": "function",
                                            "function": {"arguments": '{"command":"'},
                                        }
                                    ]
                                },
                                "finish_reason": None,
                            }
                        ],
                    },
                    {
                        "id": "chatcmpl-test",
                        "object": "chat.completion.chunk",
                        "choices": [
                            {
                                "index": 0,
                                "delta": {
                                    "tool_calls": [
                                        {
                                            "index": 0,
                                            "id": "call_1",
                                            "type": "function",
                                            "function": {"arguments": 'echo ok"}'},
                                        }
                                    ]
                                },
                                "finish_reason": None,
                            }
                        ],
                    },
                    {
                        "id": "chatcmpl-test",
                        "object": "chat.completion.chunk",
                        "choices": [{"index": 0, "delta": {}, "finish_reason": "tool_calls"}],
                    },
                ]:
                    self.wfile.write(f"data: {json.dumps(event)}\n\n".encode())
                    self.wfile.flush()
                self.wfile.write(b"data: [DONE]\n\n")
                self.wfile.flush()

            def log_message(self, *args):
                pass

        server = ThreadingHTTPServer(("127.0.0.1", 0), ToolCallStream)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            model = LitellmModel(
                model_name="openai/gpt-4o",
                model_kwargs={
                    "api_key": "fake-key",
                    "api_base": f"http://127.0.0.1:{server.server_port}/v1",
                    "stream": True,
                },
            )
            result = model.query([{"role": "user", "content": "test"}])
            assert result["extra"]["actions"] == [{"command": "echo ok", "tool_call_id": "call_1"}]
            assert result["extra"]["cost"] == 0.0
        finally:
            server.shutdown()
            server.server_close()

    @patch("minisweagent.models.litellm_model.litellm.completion")
    @patch("minisweagent.models.litellm_model.litellm.cost_calculator.completion_cost")
    def test_parse_actions_valid_tool_call(self, mock_cost, mock_completion):
        tool_call = MagicMock()
        tool_call.function.name = "bash"
        tool_call.function.arguments = '{"command": "ls -la"}'
        tool_call.id = "call_abc"
        mock_completion.return_value = _mock_litellm_response([tool_call])
        mock_cost.return_value = 0.001

        model = LitellmModel(model_name="gpt-4")
        result = model.query([{"role": "user", "content": "list files"}])
        assert result["extra"]["actions"] == [{"command": "ls -la", "tool_call_id": "call_abc"}]

    @patch("minisweagent.models.litellm_model.litellm.completion")
    @patch("minisweagent.models.litellm_model.litellm.cost_calculator.completion_cost")
    def test_parse_actions_no_tool_calls_raises(self, mock_cost, mock_completion):
        mock_completion.return_value = _mock_litellm_response(None)
        mock_cost.return_value = 0.001

        model = LitellmModel(model_name="gpt-4")
        with pytest.raises(FormatError):
            model.query([{"role": "user", "content": "test"}])

    @patch("minisweagent.models.litellm_model.litellm.completion")
    @patch("minisweagent.models.litellm_model.litellm.cost_calculator.completion_cost")
    def test_finish_reason_threaded_into_format_error_template(self, mock_cost, mock_completion):
        """The response finish_reason is exposed to format_error_template via template_kwargs, so a
        config can report a max_tokens truncation instead of the misleading "no tool call" error."""
        response = _mock_litellm_response(None)
        response.choices[0].finish_reason = "length"
        mock_completion.return_value = response
        mock_cost.return_value = 0.001

        model = LitellmModel(
            model_name="gpt-4",
            format_error_template="{% if finish_reason == 'length' %}cut off{% else %}{{ error }}{% endif %}",
        )
        with pytest.raises(FormatError) as exc:
            model.query([{"role": "user", "content": "test"}])
        assert exc.value.messages[0]["content"] == "cut off"

    def test_format_observation_messages(self):
        model = LitellmModel(model_name="gpt-4", observation_template="{{ output.output }}")
        message = {"extra": {"actions": [{"command": "echo test", "tool_call_id": "call_1"}]}}
        outputs = [{"output": "test output", "returncode": 0}]
        result = model.format_observation_messages(message, outputs)
        assert len(result) == 1
        assert result[0]["role"] == "tool"
        assert result[0]["tool_call_id"] == "call_1"
        assert result[0]["content"] == "test output"

    def test_format_observation_messages_no_actions(self):
        model = LitellmModel(model_name="gpt-4")
        result = model.format_observation_messages({"extra": {}}, [])
        assert result == []

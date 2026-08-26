from typing import Any

from minisweagent import TextGenerationResult
from minisweagent.models.utils.content_string import get_content_string


def request_kwargs(model_kwargs: dict, kwargs: dict, tools: list[dict] | None) -> dict:
    result = model_kwargs | kwargs
    if tools is None:
        result.pop("tools", None)
    else:
        result["tools"] = tools
    return result


def serialize_response(response: Any) -> Any:
    if hasattr(response, "model_dump"):
        return response.model_dump(mode="json")
    return dict(response) if not isinstance(response, dict) else response


def chat_text(response: Any) -> str:
    message = response["choices"][0]["message"] if isinstance(response, dict) else response.choices[0].message
    return get_content_string(message if isinstance(message, dict) else message.model_dump())


def responses_text(response: Any) -> str:
    if text := getattr(response, "output_text", None):
        return text
    serialized = serialize_response(response)
    return serialized.get("output_text") or get_content_string(serialized)


def responses_kwargs(kwargs: dict[str, Any]) -> dict[str, Any]:
    kwargs = dict(kwargs)
    if "max_tokens" in kwargs:
        kwargs.setdefault("max_output_tokens", kwargs.pop("max_tokens"))
    return kwargs


def text_generation_result(text: str, response: Any, cost: float) -> TextGenerationResult:
    return {"text": text, "cost": cost, "response": serialize_response(response)}

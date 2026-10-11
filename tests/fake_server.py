"""A fake Anthropic server for the conformance kit: the real ``anthropic`` SDK talks to it through
``httpx.MockTransport``, so requests and answers go through the SDK's own encoding and parsing."""

from __future__ import annotations

import json
from collections import deque

import httpx
from marvin_integration_sdk.ai import ToolCall
from marvin_integration_sdk.ai.fake import FakeTransport


class FakeAnthropicServer(FakeTransport):
    """Queued neutral replies, answered in the Messages API's wire shape (``POST /v1/messages``,
    ``GET /v1/models``)."""

    def __init__(self) -> None:
        self.replies: deque[dict] = deque()
        self.requests: list[httpx.Request] = []

    # ── the kit's vocabulary ─────────────────────────────────────────────────

    def reply_text(self, text: str, *, prompt_tokens: int = 11, completion_tokens: int = 7) -> None:
        self.replies.append({"text": text, "usage": (prompt_tokens, completion_tokens)})

    def reply_tool_calls(self, calls: list[ToolCall], *, prompt_tokens: int = 11, completion_tokens: int = 7) -> None:
        self.replies.append({"tool_calls": list(calls), "usage": (prompt_tokens, completion_tokens)})

    def reply_models(self, ids: list[str]) -> None:
        self.replies.append({"models": list(ids)})

    def reply_error(self, status: int, message: str) -> None:
        self.replies.append({"error": (status, message)})

    def reply_connection_ok(self) -> None:
        # test_connection sends a one-token message, not a model listing.
        self.reply_text("hi", prompt_tokens=8, completion_tokens=1)

    def last_request(self) -> str:
        return self.requests[-1].content.decode() if self.requests else ""

    def last_json(self) -> dict:
        return json.loads(self.last_request() or "{}")

    def client(self) -> httpx.Client:
        return httpx.Client(transport=httpx.MockTransport(self._handle))

    # ── the server ───────────────────────────────────────────────────────────

    def _handle(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        reply = self.replies.popleft() if self.replies else {}
        if "error" in reply:
            status, message = reply["error"]
            return httpx.Response(status, json={"type": "error", "error": {"type": _error_type(status), "message": message}})
        path = request.url.path
        if request.method == "POST" and path == "/v1/messages":
            return httpx.Response(200, json=_message(reply, json.loads(request.content or b"{}")))
        if request.method == "GET" and path == "/v1/models":
            data = [{"type": "model", "id": i, "display_name": i, "created_at": "2026-01-01T00:00:00Z"} for i in reply.get("models", [])]
            ids = [d["id"] for d in data]
            return httpx.Response(
                200, json={"data": data, "has_more": False, "first_id": ids[0] if ids else None, "last_id": ids[-1] if ids else None}
            )
        return httpx.Response(404, json={"type": "error", "error": {"type": "not_found_error", "message": f"no fake for {path}"}})


def _error_type(status: int) -> str:
    return {400: "invalid_request_error", 401: "authentication_error", 403: "permission_error", 404: "not_found_error", 429: "rate_limit_error"}.get(
        status, "api_error"
    )


def _message(reply: dict, body: dict) -> dict:
    """The reply as a Messages API response. A forced tool call (structured output) is answered the way
    the real API answers it: with a ``tool_use`` block whose input is the JSON the script gave as text."""
    pt, ct = reply.get("usage", (0, 0))
    content: list[dict] = []
    text = reply.get("text")
    forced = body.get("tool_choice", {}).get("type") == "tool"
    if forced and text is not None:
        content.append({"type": "tool_use", "id": "toolu_forced", "name": body["tool_choice"]["name"], "input": json.loads(text)})
    elif text is not None:
        content.append({"type": "text", "text": text})
    for call in reply.get("tool_calls", []):
        content.append({"type": "tool_use", "id": call.id, "name": call.name, "input": call.arguments})
    used_tools = any(block["type"] == "tool_use" for block in content)
    return {
        "id": "msg_1",
        "type": "message",
        "role": "assistant",
        "model": body.get("model", ""),
        "content": content,
        "stop_reason": "tool_use" if used_tools else "end_turn",
        "stop_sequence": None,
        "usage": {"input_tokens": pt, "output_tokens": ct},
    }

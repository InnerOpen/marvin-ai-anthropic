"""Anthropic's Messages API: completions, tool calling, structured output (a forced tool) and vision."""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any

from marvin_integration_sdk.ai import (
    API_KEY,
    AIProvider,
    CompletionOptions,
    CompletionResult,
    ImagePart,
    Message,
    ToolCall,
    ToolDefinition,
)

from .prices import ANTHROPIC_PRICES

DEFAULT_MAX_TOKENS = 4096
STRUCTURED_TOOL = "structured_output"
# The cheapest current model: test_connection spends one token on it.
CONNECTION_TEST_MODEL = "claude-haiku-4-5-20251001"
TOOL_CHOICES = {"auto": {"type": "auto"}, "required": {"type": "any"}, "none": {"type": "none"}}


def _sampling(opts: CompletionOptions) -> dict:
    """A temperature only when one is configured; otherwise the model's default."""
    return {"temperature": opts.temperature} if opts.temperature is not None else {}


def _result(resp, **extra) -> CompletionResult:
    return CompletionResult(
        content=extra.pop("content", ""),
        prompt_tokens=resp.usage.input_tokens,
        completion_tokens=resp.usage.output_tokens,
        total_tokens=resp.usage.input_tokens + resp.usage.output_tokens,
        model=resp.model,
        **extra,
    )


class AnthropicProvider(AIProvider):
    provider_type = "anthropic"
    display_name = "Anthropic"
    supports_vision = True
    supports_structured_output = True
    supports_tool_calls = True
    credentials = (API_KEY,)
    prices = ANTHROPIC_PRICES
    default_model = "claude-sonnet-5"
    suggested_models = tuple(ANTHROPIC_PRICES)

    def __init__(self, api_key: str, base_url: str | None = None, http_client=None, max_retries: int = 2) -> None:
        """``base_url`` points at a proxy or gateway; ``http_client`` and ``max_retries`` are how tests inject
        a transport (the SDK's own default retries are 2)."""
        self._api_key = api_key
        self._base_url = base_url
        self._http_client = http_client
        self._max_retries = max_retries

    @classmethod
    def from_credentials(cls, values: Mapping[str, Any]) -> AnthropicProvider:
        return cls(api_key=values.get("api_key") or "")

    def _client(self):
        try:
            import anthropic
        except ImportError:
            raise ImportError("The Anthropic SDK is not installed (pip install marvin-ai-anthropic brings it).") from None
        return anthropic.Anthropic(api_key=self._api_key, base_url=self._base_url, http_client=self._http_client, max_retries=self._max_retries)

    def _render_content(self, content):
        """Translate agnostic content into Anthropic's content format (str or content blocks)."""
        if isinstance(content, str):
            return content
        blocks = []
        for p in content:
            if isinstance(p, ImagePart):
                blocks.append({"type": "image", "source": {"type": "base64", "media_type": p.mime_type, "data": p.data}})
            else:
                blocks.append({"type": "text", "text": str(p)})
        return blocks

    def _split_messages(self, messages: list[Message]):
        system = " ".join(m.content for m in messages if m.role == "system" and isinstance(m.content, str))
        chat = [{"role": m.role, "content": self._render_content(m.content)} for m in messages if m.role != "system"]
        return system or None, chat

    def _split_tool_messages(self, messages: list[Message]):
        """Like _split_messages, but renders the tool-calling round-trip into Anthropic's blocks.

        assistant tool calls become `tool_use` blocks; role="tool" results become `tool_result`
        blocks in a user message (consecutive results are merged into one user turn, as Anthropic
        expects the results immediately following the assistant's tool_use).
        """
        system = " ".join(m.content for m in messages if m.role == "system" and isinstance(m.content, str))
        chat: list[dict] = []
        pending_results: list[dict] = []

        def flush_results() -> None:
            if pending_results:
                chat.append({"role": "user", "content": list(pending_results)})
                pending_results.clear()

        for m in messages:
            if m.role == "system":
                continue
            if m.role == "tool":
                content = m.content if isinstance(m.content, str) else self._render_content(m.content)
                pending_results.append({"type": "tool_result", "tool_use_id": m.tool_call_id, "content": content})
                continue
            flush_results()
            if m.role == "assistant" and m.tool_calls:
                blocks: list[dict] = []
                if isinstance(m.content, str) and m.content.strip():
                    blocks.append({"type": "text", "text": m.content})
                elif isinstance(m.content, list):
                    blocks.extend(self._render_content(m.content))
                for tc in m.tool_calls:
                    blocks.append({"type": "tool_use", "id": tc.id, "name": tc.name, "input": tc.arguments})
                chat.append({"role": "assistant", "content": blocks})
            else:
                chat.append({"role": m.role, "content": self._render_content(m.content)})
        flush_results()
        return system or None, chat

    def complete(self, messages: list[Message], model: str, options: CompletionOptions | None = None) -> CompletionResult:
        opts = options or CompletionOptions()
        system, chat = self._split_messages(messages)
        kwargs = {"model": model, "messages": chat, "max_tokens": opts.max_tokens or DEFAULT_MAX_TOKENS, **_sampling(opts)}
        if system:
            kwargs["system"] = system
        resp = self._client().messages.create(**kwargs)
        return _result(resp, content=resp.content[0].text if resp.content else "", raw=resp.model_dump())

    def complete_with_tools(
        self,
        messages: list[Message],
        model: str,
        tools: list[ToolDefinition],
        options: CompletionOptions | None = None,
        tool_choice: str = "auto",
    ) -> CompletionResult:
        opts = options or CompletionOptions()
        system, chat = self._split_tool_messages(messages)
        kwargs = {
            "model": model,
            "messages": chat,
            "max_tokens": opts.max_tokens or DEFAULT_MAX_TOKENS,
            **_sampling(opts),
            "tools": [{"name": t.name, "description": t.description, "input_schema": t.input_schema} for t in tools],
            "tool_choice": TOOL_CHOICES.get(tool_choice, {"type": "auto"}),
        }
        if system:
            kwargs["system"] = system
        resp = self._client().messages.create(**kwargs)

        text_parts: list[str] = []
        tool_calls: list[ToolCall] = []
        for block in resp.content:
            btype = getattr(block, "type", None)
            if btype == "text":
                text_parts.append(block.text)
            elif btype == "tool_use":
                tool_calls.append(ToolCall(id=block.id, name=block.name, arguments=dict(block.input or {})))
        return _result(resp, content="".join(text_parts), raw=resp.model_dump(), tool_calls=tool_calls, stop_reason=resp.stop_reason)

    def _structured_response(self, messages: list[Message], model: str, output_schema: dict, options: CompletionOptions | None):
        """Structured output is a forced call of one tool whose input schema is the output schema."""
        opts = options or CompletionOptions()
        system, chat = self._split_messages(messages)
        tool = {"name": STRUCTURED_TOOL, "description": "Return structured output", "input_schema": output_schema}
        kwargs = {
            "model": model,
            "messages": chat,
            "max_tokens": opts.max_tokens or DEFAULT_MAX_TOKENS,
            "tools": [tool],
            "tool_choice": {"type": "tool", "name": STRUCTURED_TOOL},
        }
        if system:
            kwargs["system"] = system
        return self._client().messages.create(**kwargs)

    def complete_structured(self, messages: list[Message], model: str, output_schema: dict, options: CompletionOptions | None = None) -> dict:
        resp = self._structured_response(messages, model, output_schema, options)
        for block in resp.content:
            if hasattr(block, "input"):
                return block.input
        return json.loads(resp.content[0].text if resp.content else "{}")

    def execute_operation(self, messages, model, output_schema, options=None):
        resp = self._structured_response(messages, model, output_schema, options)
        parsed = {}
        for block in resp.content:
            if hasattr(block, "input"):
                parsed = block.input
                break
        return parsed, _result(resp, content=str(parsed))

    def list_models(self) -> list[str]:
        # limit=1000 is the API's maximum page; iterating the page follows any further ones.
        return sorted(m.id for m in self._client().models.list(limit=1000))

    def test_connection(self) -> tuple[bool, str]:
        try:
            self._client().messages.create(model=CONNECTION_TEST_MODEL, messages=[{"role": "user", "content": "hi"}], max_tokens=1)
            return True, "Connected"
        except Exception as e:  # noqa: BLE001 — reported to the admin, never raised
            return False, str(e)

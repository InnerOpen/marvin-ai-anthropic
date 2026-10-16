"""Anthropic's Messages API: completions, tool calling, structured output (a forced tool) and vision."""

from __future__ import annotations

import json
import logging
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
FORCED_CHOICES = ("any", "tool")

logger = logging.getLogger(__name__)

# Models that answered a forced tool choice (`any`, or one named `tool`) with 400 "not supported for this model"
# (claude-sonnet-5-5 does). They get `auto` plus an instruction to call the tool, without trying the forced call first.
_UNFORCED_MODELS: set[str] = set()


def _refuses_forced_choice(error: Exception) -> bool:
    import anthropic

    message = f"{error} {getattr(error, 'body', '')}"  # the API's message is in both, depending on the SDK version
    return isinstance(error, anthropic.BadRequestError) and "tool_choice" in message and "not supported" in message


def _unforced(kwargs: dict, instruction: str) -> dict:
    """The same request with tool_choice auto and the instruction appended to the system prompt (its own block, so
    the cached one stays as it was)."""
    return {**kwargs, "tool_choice": {"type": "auto"}, "system": [*kwargs.get("system", []), {"type": "text", "text": instruction}]}


def _sampling(opts: CompletionOptions) -> dict:
    """A temperature only when one is configured; otherwise the model's default."""
    return {"temperature": opts.temperature} if opts.temperature is not None else {}


CACHE_MARK = {"type": "ephemeral"}


def _result(resp, **extra) -> CompletionResult:
    # Anthropic's input_tokens leaves out what its prompt cache served or stored; Marvin's prompt_tokens is the
    # whole prompt, with those two as its parts.
    usage = resp.usage
    cached = getattr(usage, "cache_read_input_tokens", None) or 0
    written = getattr(usage, "cache_creation_input_tokens", None) or 0
    prompt = usage.input_tokens + cached + written
    return CompletionResult(
        content=extra.pop("content", ""),
        prompt_tokens=prompt,
        completion_tokens=usage.output_tokens,
        total_tokens=prompt + usage.output_tokens,
        model=resp.model,
        cached_prompt_tokens=cached,
        cache_write_tokens=written,
        **extra,
    )


def _text(resp) -> str:
    """The answer's text: its text blocks joined. A model may put a thinking block first (claude-sonnet-5-5 does)."""
    return "".join(b.text for b in resp.content if getattr(b, "type", None) == "text")


def _parsed(resp) -> dict:
    """A structured answer: the tool call's input, else JSON in the text (a model asked rather than forced to call
    the tool may answer in text, possibly fenced as a code block)."""
    for block in resp.content:
        if getattr(block, "type", None) == "tool_use":
            return dict(block.input or {})
    text = _text(resp).strip()
    start, end = text.find("{"), text.rfind("}")
    return json.loads(text[start : end + 1]) if start != -1 and end > start else json.loads(text or "{}")


def _system_blocks(system: str) -> list[dict]:
    """The system prompt as one cached block: the persona and the workspace preamble are the same on every call of a
    conversation, so after the first they are read from the prompt cache at a tenth of the price."""
    return [{"type": "text", "text": system, "cache_control": CACHE_MARK}]


def _tool_blocks(tools: list[ToolDefinition]) -> list[dict]:
    """The tool definitions, the last one marked so the whole list (the bulk of an agent's prompt) is cached."""
    out = [{"name": t.name, "description": t.description, "input_schema": t.input_schema} for t in tools]
    if out:
        out[-1] = {**out[-1], "cache_control": CACHE_MARK}
    return out


class AnthropicProvider(AIProvider):
    provider_type = "anthropic"
    logo = "logo.svg"  # the Claude mark: Marvin shows it beside the provider's name
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
            kwargs["system"] = _system_blocks(system)
        resp = self._client().messages.create(**kwargs)
        return _result(resp, content=_text(resp), raw=resp.model_dump())

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
            "tools": _tool_blocks(tools),
            "tool_choice": TOOL_CHOICES.get(tool_choice, {"type": "auto"}),
        }
        if system:
            kwargs["system"] = _system_blocks(system)
        resp = self._create(kwargs, "Answer by calling one of the tools.")

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
            kwargs["system"] = _system_blocks(system)
        return self._create(kwargs, f"Answer by calling the {STRUCTURED_TOOL} tool with your answer as its input.")

    def _create(self, kwargs: dict, instruction: str):
        """messages.create. A forced tool choice that the model refuses is sent again as `auto` with `instruction`,
        and the model is remembered, so its later calls go straight to that."""
        client = self._client()
        if kwargs.get("tool_choice", {}).get("type") not in FORCED_CHOICES:
            return client.messages.create(**kwargs)
        model = kwargs["model"]
        if model not in _UNFORCED_MODELS:
            try:
                return client.messages.create(**kwargs)
            except Exception as e:
                if not _refuses_forced_choice(e):
                    raise
                _UNFORCED_MODELS.add(model)
                logger.warning("%s refuses a forced tool choice; asking it to call the tool instead", model)
        return client.messages.create(**_unforced(kwargs, instruction))

    def complete_structured(self, messages: list[Message], model: str, output_schema: dict, options: CompletionOptions | None = None) -> dict:
        return _parsed(self._structured_response(messages, model, output_schema, options))

    def execute_operation(self, messages, model, output_schema, options=None):
        resp = self._structured_response(messages, model, output_schema, options)
        parsed = _parsed(resp)
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

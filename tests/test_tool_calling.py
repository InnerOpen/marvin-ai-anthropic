"""Tool calling and message translation for the Messages API (ported from Marvin core's
tests/test_provider_tool_calling.py). The SDK client is mocked: no network, no key."""

import json
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from marvin_integration_sdk.ai import CompletionOptions, ImagePart, Message, ToolCall, ToolDefinition

from marvin_ai_anthropic import AnthropicProvider

TOOLS = [ToolDefinition(name="get_entry", description="Fetch an entry", input_schema={"type": "object"})]


def _response(content, stop_reason="end_turn", usage=(12, 6)):
    return SimpleNamespace(
        content=content,
        usage=SimpleNamespace(input_tokens=usage[0], output_tokens=usage[1]),
        model="claude-sonnet-5",
        stop_reason=stop_reason,
        model_dump=dict,
    )


def _with_client(provider, response):
    client = MagicMock()
    client.messages.create.return_value = response
    provider._client = MagicMock(return_value=client)
    return client


def test_capability_flags():
    assert AnthropicProvider.supports_tool_calls is True
    assert AnthropicProvider.supports_vision is True
    assert AnthropicProvider.supports_structured_output is True


def test_splits_tool_roundtrip():
    provider = AnthropicProvider(api_key="x")
    messages = [
        Message(role="system", content="You are Marvin."),
        Message(role="user", content="fetch about"),
        Message(role="assistant", content="Let me look.", tool_calls=[ToolCall(id="tu_1", name="get_entry", arguments={"slug": "about"})]),
        Message(role="tool", content='{"title":"About"}', tool_call_id="tu_1"),
    ]
    system, chat = provider._split_tool_messages(messages)

    assert system == "You are Marvin."
    # assistant turn carries a text block + a tool_use block
    assistant = chat[1]
    assert assistant["role"] == "assistant"
    assert [b["type"] for b in assistant["content"]] == ["text", "tool_use"]
    assert assistant["content"][1] == {"type": "tool_use", "id": "tu_1", "name": "get_entry", "input": {"slug": "about"}}
    # tool result becomes a user message with a tool_result block
    result_turn = chat[2]
    assert result_turn["role"] == "user"
    assert result_turn["content"][0] == {"type": "tool_result", "tool_use_id": "tu_1", "content": '{"title":"About"}'}


def test_a_tool_call_without_text_has_no_text_block():
    provider = AnthropicProvider(api_key="x")
    _, chat = provider._split_tool_messages([Message(role="assistant", content="  ", tool_calls=[ToolCall(id="a", name="t", arguments={})])])
    assert [b["type"] for b in chat[0]["content"]] == ["tool_use"]


def test_merges_consecutive_tool_results():
    provider = AnthropicProvider(api_key="x")
    messages = [
        Message(role="assistant", content="", tool_calls=[ToolCall(id="a", name="t", arguments={}), ToolCall(id="b", name="t", arguments={})]),
        Message(role="tool", content="ra", tool_call_id="a"),
        Message(role="tool", content="rb", tool_call_id="b"),
    ]
    _system, chat = provider._split_tool_messages(messages)
    # both results collapse into a single user turn with two tool_result blocks
    result_turn = chat[1]
    assert result_turn["role"] == "user"
    assert [b["tool_use_id"] for b in result_turn["content"]] == ["a", "b"]


def test_an_image_becomes_a_base64_block():
    provider = AnthropicProvider(api_key="x")
    _, chat = provider._split_messages([Message("user", ["What is this?", ImagePart(data="QUJD", mime_type="image/png")])])
    assert chat[0]["content"] == [
        {"type": "text", "text": "What is this?"},
        {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": "QUJD"}},
    ]


def test_complete_with_tools_parses_tool_use():
    provider = AnthropicProvider(api_key="x")
    client = _with_client(
        provider,
        _response(
            [SimpleNamespace(type="text", text="Looking… "), SimpleNamespace(type="tool_use", id="tu_9", name="get_entry", input={"slug": "home"})],
            stop_reason="tool_use",
        ),
    )

    result = provider.complete_with_tools([Message(role="user", content="hi")], "claude-sonnet-5", TOOLS)

    assert result.content == "Looking… "
    assert result.stop_reason == "tool_use"
    assert result.tool_calls == [ToolCall(id="tu_9", name="get_entry", arguments={"slug": "home"})]
    assert result.total_tokens == 18
    sent = client.messages.create.call_args.kwargs
    assert sent["tools"][0]["name"] == "get_entry"
    assert sent["tool_choice"] == {"type": "auto"}


@pytest.mark.parametrize(("choice", "expected"), [("required", {"type": "any"}), ("none", {"type": "none"}), ("bogus", {"type": "auto"})])
def test_tool_choice_maps(choice, expected):
    provider = AnthropicProvider(api_key="x")
    client = _with_client(provider, _response([SimpleNamespace(type="text", text="x")]))
    provider.complete_with_tools([Message(role="user", content="hi")], "claude-sonnet-5", TOOLS, tool_choice=choice)
    assert client.messages.create.call_args.kwargs["tool_choice"] == expected


def test_system_text_and_options_reach_the_request():
    provider = AnthropicProvider(api_key="x")
    client = _with_client(provider, _response([SimpleNamespace(type="text", text="ok")]))
    provider.complete([Message("system", "Be brief."), Message("user", "hi")], "claude-sonnet-5", CompletionOptions(max_tokens=50, temperature=0.2))
    sent = client.messages.create.call_args.kwargs
    assert sent["system"] == [{"type": "text", "text": "Be brief.", "cache_control": {"type": "ephemeral"}}]
    assert sent["max_tokens"] == 50 and sent["temperature"] == 0.2
    assert sent["messages"] == [{"role": "user", "content": "hi"}]


def test_no_temperature_is_sent_unless_configured():
    provider = AnthropicProvider(api_key="x")
    client = _with_client(provider, _response([SimpleNamespace(type="text", text="ok")]))
    provider.complete([Message("user", "hi")], "claude-sonnet-5")
    sent = client.messages.create.call_args.kwargs
    assert "temperature" not in sent and "system" not in sent


def test_structured_output_forces_one_tool():
    provider = AnthropicProvider(api_key="x")
    schema = {"type": "object", "properties": {"tags": {"type": "array"}}}
    client = _with_client(provider, _response([SimpleNamespace(type="tool_use", id="t", name="structured_output", input={"tags": ["a"]})]))

    assert provider.complete_structured([Message("user", "Tag this.")], "claude-sonnet-5", schema) == {"tags": ["a"]}
    sent = client.messages.create.call_args.kwargs
    assert sent["tool_choice"] == {"type": "tool", "name": "structured_output"}
    assert sent["tools"][0]["input_schema"] == schema


def test_structured_output_falls_back_to_json_text():
    provider = AnthropicProvider(api_key="x")
    _with_client(provider, _response([SimpleNamespace(type="text", text=json.dumps({"a": 1}))]))
    assert provider.complete_structured([Message("user", "x")], "claude-sonnet-5", {"type": "object"}) == {"a": 1}


def test_execute_operation_returns_the_parsed_answer_and_usage():
    provider = AnthropicProvider(api_key="x")
    _with_client(provider, _response([SimpleNamespace(type="tool_use", id="t", name="structured_output", input={"summary": "s"})], usage=(30, 5)))
    parsed, result = provider.execute_operation([Message("user", "Summarise.")], "claude-sonnet-5", {"type": "object"})
    assert parsed == {"summary": "s"}
    assert (result.prompt_tokens, result.completion_tokens, result.total_tokens) == (30, 5, 35)


def test_the_client_is_built_from_the_constructor():
    client = AnthropicProvider(api_key="k", base_url="http://proxy.invalid", max_retries=0)._client()
    assert client.api_key == "k" and str(client.base_url).startswith("http://proxy.invalid") and client.max_retries == 0
    assert AnthropicProvider.from_credentials({"api_key": "k"})._max_retries == 2


def test_the_system_and_the_tools_are_marked_for_the_prompt_cache():
    """The persona and the tool list are the same on every call of a conversation: one cache mark on the system
    block and one on the last tool caches that whole prefix."""
    client = MagicMock()
    client.messages.create.return_value = SimpleNamespace(
        content=[], stop_reason="end_turn", model="claude-sonnet-5", usage=SimpleNamespace(input_tokens=5, output_tokens=1), model_dump=dict
    )
    provider = AnthropicProvider(api_key="x")
    provider._client = lambda: client
    two = [*TOOLS, ToolDefinition(name="list_assets", description="List assets", input_schema={"type": "object"})]
    provider.complete_with_tools([Message("system", "Be brief."), Message("user", "hi")], "claude-sonnet-5", two)

    sent = client.messages.create.call_args.kwargs
    assert sent["system"] == [{"type": "text", "text": "Be brief.", "cache_control": {"type": "ephemeral"}}]
    assert sent["tools"][-1]["cache_control"] == {"type": "ephemeral"} and "cache_control" not in sent["tools"][0]


def _refusal():
    import anthropic
    import httpx

    request = httpx.Request("POST", "https://api.anthropic.com/v1/messages")
    body = {
        "type": "error",
        "error": {"type": "invalid_request_error", "message": 'tool_choice: type "tool" and "any" are not supported for this model.'},
    }
    return anthropic.BadRequestError("Error code: 400", response=httpx.Response(400, request=request, json=body), body=body)


def test_a_model_that_refuses_a_forced_tool_is_asked_to_call_it_instead(monkeypatch):
    from marvin_ai_anthropic import provider as mod

    monkeypatch.setattr(mod, "_UNFORCED_MODELS", set())
    provider = AnthropicProvider(api_key="x")
    client = _with_client(provider, None)
    answer = _response([SimpleNamespace(type="tool_use", id="t", name="structured_output", input={"tags": ["a"]})])
    client.messages.create.side_effect = [_refusal(), answer, answer]

    assert provider.complete_structured([Message("system", "Tag."), Message("user", "x")], "claude-sonnet-5-5", {"type": "object"}) == {"tags": ["a"]}
    forced, unforced = (c.kwargs for c in client.messages.create.call_args_list)
    assert forced["tool_choice"] == {"type": "tool", "name": "structured_output"}
    assert unforced["tool_choice"] == {"type": "auto"}
    assert unforced["system"][0] == forced["system"][0]  # the cached block is untouched
    assert "structured_output" in unforced["system"][-1]["text"]

    # Remembered: the next call goes straight to auto.
    provider.complete_structured([Message("user", "y")], "claude-sonnet-5-5", {"type": "object"})
    assert client.messages.create.call_count == 3 and client.messages.create.call_args.kwargs["tool_choice"] == {"type": "auto"}


def test_a_required_tool_call_falls_back_the_same_way(monkeypatch):
    from marvin_ai_anthropic import provider as mod

    monkeypatch.setattr(mod, "_UNFORCED_MODELS", set())
    provider = AnthropicProvider(api_key="x")
    client = _with_client(provider, None)
    call = _response([SimpleNamespace(type="tool_use", id="tu", name="get_entry", input={})], stop_reason="tool_use")
    client.messages.create.side_effect = [_refusal(), call]

    result = provider.complete_with_tools([Message("user", "hi")], "claude-sonnet-5-5", TOOLS, tool_choice="required")
    assert [c.name for c in result.tool_calls] == ["get_entry"]
    assert client.messages.create.call_args.kwargs["tool_choice"] == {"type": "auto"}


def test_other_bad_requests_are_not_retried(monkeypatch):
    import anthropic
    import httpx

    from marvin_ai_anthropic import provider as mod

    monkeypatch.setattr(mod, "_UNFORCED_MODELS", set())
    provider = AnthropicProvider(api_key="x")
    client = _with_client(provider, None)
    request = httpx.Request("POST", "https://api.anthropic.com/v1/messages")
    client.messages.create.side_effect = anthropic.BadRequestError("bad", response=httpx.Response(400, request=request), body=None)
    with pytest.raises(anthropic.BadRequestError):
        provider.complete_structured([Message("user", "x")], "claude-sonnet-5-5", {"type": "object"})
    assert client.messages.create.call_count == 1 and mod._UNFORCED_MODELS == set()


def test_a_structured_answer_in_fenced_text_is_read():
    provider = AnthropicProvider(api_key="x")
    _with_client(provider, _response([SimpleNamespace(type="text", text='Here it is:\n```json\n{"a": 1}\n```')]))
    assert provider.complete_structured([Message("user", "x")], "claude-sonnet-5", {"type": "object"}) == {"a": 1}

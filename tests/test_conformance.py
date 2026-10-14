"""The SDK's AI provider conformance kit, over a fake Anthropic server."""

import pytest
from fake_server import FakeAnthropicServer
from marvin_integration_sdk.ai import Message
from marvin_integration_sdk.ai.testing import AIProviderContract

from marvin_ai_anthropic import AnthropicProvider, plugin

PLACEHOLDER_KEY = "test-key"


class TestAnthropic(AIProviderContract):
    @pytest.fixture
    def transport(self):
        return FakeAnthropicServer()

    @pytest.fixture
    def provider(self, transport):
        return AnthropicProvider(api_key=PLACEHOLDER_KEY, http_client=transport.client(), max_retries=0)

    def test_it_talks_to_the_messages_api(self, provider, transport, model):
        transport.reply_text("hi")
        provider.complete([Message("user", "hi")], model)
        request = transport.requests[-1]
        assert request.url.path == "/v1/messages"
        assert request.headers["x-api-key"] == PLACEHOLDER_KEY
        assert transport.last_json()["max_tokens"] == 4096

    def test_the_model_listing_is_the_accounts(self, provider, transport):
        transport.reply_models(["claude-b", "claude-a"])
        assert provider.list_models() == ["claude-a", "claude-b"]
        assert transport.requests[-1].url.path == "/v1/models"


def test_the_entry_point_declares_the_plugin():
    from importlib.metadata import entry_points

    eps = {ep.name: ep.load() for ep in entry_points(group="marvin.ai_providers")}
    assert eps["anthropic"] is plugin


def test_cache_reads_and_writes_are_parts_of_the_prompt():
    """Anthropic's input_tokens leaves the cached and written parts out; Marvin's prompt_tokens is the whole prompt."""
    transport = FakeAnthropicServer()
    transport.reply_text("ok", prompt_tokens=100, completion_tokens=3, cached=25000, written=2000)
    provider = AnthropicProvider(api_key=PLACEHOLDER_KEY, http_client=transport.client(), max_retries=0)
    result = provider.complete([Message("user", "hi")], "claude-sonnet-5")
    assert (result.prompt_tokens, result.cached_prompt_tokens, result.cache_write_tokens, result.total_tokens) == (27100, 25000, 2000, 27103)

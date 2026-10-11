"""One optional smoke test against the real Anthropic API. Skipped unless ANTHROPIC_API_KEY is set; CI has none."""

import os

import pytest
from marvin_integration_sdk.ai import CompletionOptions, Message

from marvin_ai_anthropic import AnthropicProvider

pytestmark = pytest.mark.live


@pytest.mark.skipif(not os.environ.get("ANTHROPIC_API_KEY"), reason="ANTHROPIC_API_KEY is not set")
def test_a_real_completion():
    provider = AnthropicProvider(api_key=os.environ["ANTHROPIC_API_KEY"])
    result = provider.complete(
        [Message("user", "Reply with the single word: towel")],
        os.environ.get("ANTHROPIC_MODEL", "claude-haiku-4-5"),
        CompletionOptions(max_tokens=16),
    )
    assert result.content.strip() and result.prompt_tokens > 0

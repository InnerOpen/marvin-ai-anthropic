from marvin_ai_anthropic import AnthropicProvider
from marvin_ai_anthropic.prices import ANTHROPIC_PRICES


def test_the_table_is_the_providers():
    assert AnthropicProvider.prices is ANTHROPIC_PRICES
    assert AnthropicProvider.suggested_models == tuple(ANTHROPIC_PRICES)
    assert AnthropicProvider.default_model in ANTHROPIC_PRICES


def test_the_models_core_named_are_priced():
    for model in ("claude-opus-4-8", "claude-sonnet-5", "claude-haiku-4-5-20251001"):
        assert AnthropicProvider.price(model) is not None


def test_a_dated_snapshot_takes_its_base_price_and_a_newer_family_member_does_not_fall_back():
    assert AnthropicProvider.price("claude-haiku-4-5-20251001") == ANTHROPIC_PRICES["claude-haiku-4-5"]
    assert AnthropicProvider.price("claude-sonnet-5-5") == ANTHROPIC_PRICES["claude-sonnet-5-5"]
    assert AnthropicProvider.price("claude-sonnet-9") is None

"""What Anthropic charges, per model: USD per million tokens, base input and output.

From LiteLLM's model price list (``litellm_provider: anthropic``, ``input_cost_per_token`` and
``output_cost_per_token`` × 1e6), 2026-10-10. Base ids only: a dated snapshot (``claude-haiku-4-5-20251001``)
takes the price of the id it extends; a model that isn't listed has no estimate (Marvin shows "—" and
counts the run as unpriced). A new model's price is a release of this package, not of Marvin.
"""

from marvin_integration_sdk.ai import ModelPrice

ANTHROPIC_PRICES: dict[str, ModelPrice] = {
    # Fable and Mythos.
    "claude-fable-5-1": ModelPrice(input_per_1m=10.00, output_per_1m=50.00),
    "claude-fable-5": ModelPrice(input_per_1m=10.00, output_per_1m=50.00),
    "claude-mythos-5-1": ModelPrice(input_per_1m=10.00, output_per_1m=50.00),
    "claude-mythos-5": ModelPrice(input_per_1m=10.00, output_per_1m=50.00),
    "claude-mythos-preview": ModelPrice(input_per_1m=10.00, output_per_1m=50.00),
    # Opus.
    "claude-opus-5-5": ModelPrice(input_per_1m=4.00, output_per_1m=20.00),
    "claude-opus-5": ModelPrice(input_per_1m=5.00, output_per_1m=25.00),
    "claude-opus-4-8": ModelPrice(input_per_1m=5.00, output_per_1m=25.00),
    "claude-opus-4-7": ModelPrice(input_per_1m=5.00, output_per_1m=25.00),
    "claude-opus-4-6": ModelPrice(input_per_1m=5.00, output_per_1m=25.00),
    "claude-opus-4-5": ModelPrice(input_per_1m=5.00, output_per_1m=25.00),
    # Sonnet.
    "claude-sonnet-5-5": ModelPrice(input_per_1m=2.00, output_per_1m=10.00),
    "claude-sonnet-5": ModelPrice(input_per_1m=2.00, output_per_1m=10.00),
    "claude-sonnet-4-6": ModelPrice(input_per_1m=3.00, output_per_1m=15.00),
    "claude-sonnet-4-5": ModelPrice(input_per_1m=3.00, output_per_1m=15.00),
    # Haiku.
    "claude-haiku-5-5": ModelPrice(input_per_1m=0.10, output_per_1m=0.50),
    "claude-haiku-4-5": ModelPrice(input_per_1m=1.00, output_per_1m=5.00),
}

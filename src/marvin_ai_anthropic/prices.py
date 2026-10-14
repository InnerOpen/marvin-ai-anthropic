"""What Anthropic charges, per model: USD per million tokens, base input and output.

From LiteLLM's model price list (``litellm_provider: anthropic``, ``input_cost_per_token`` and
``output_cost_per_token`` × 1e6), 2026-10-10. Prompt-cache rates follow Anthropic's rule for every current model
(``cache_read_input_token_cost`` = 0.1 × input, ``cache_creation_input_token_cost`` = 1.25 × input), so each price
is built with :func:`_price`. Base ids only: a dated snapshot (``claude-haiku-4-5-20251001``)
takes the price of the id it extends; a model that isn't listed has no estimate (Marvin shows "—" and
counts the run as unpriced). A new model's price is a release of this package, not of Marvin.
"""

from marvin_integration_sdk.ai import ModelPrice


def _price(input_per_1m: float, output_per_1m: float) -> ModelPrice:
    """A model's price with Anthropic's cache rates: reads at a tenth of the input rate, writes at 1.25×."""
    return ModelPrice(
        input_per_1m=input_per_1m,
        output_per_1m=output_per_1m,
        cached_input_per_1m=round(input_per_1m * 0.1, 6),
        cache_write_per_1m=round(input_per_1m * 1.25, 6),
    )


ANTHROPIC_PRICES: dict[str, ModelPrice] = {
    # Fable and Mythos.
    "claude-fable-5-1": _price(10.00, 50.00),
    "claude-fable-5": _price(10.00, 50.00),
    "claude-mythos-5-1": _price(10.00, 50.00),
    "claude-mythos-5": _price(10.00, 50.00),
    "claude-mythos-preview": _price(10.00, 50.00),
    # Opus.
    "claude-opus-5-5": _price(4.00, 20.00),
    "claude-opus-5": _price(5.00, 25.00),
    "claude-opus-4-8": _price(5.00, 25.00),
    "claude-opus-4-7": _price(5.00, 25.00),
    "claude-opus-4-6": _price(5.00, 25.00),
    "claude-opus-4-5": _price(5.00, 25.00),
    # Sonnet.
    "claude-sonnet-5-5": _price(2.00, 10.00),
    "claude-sonnet-5": _price(2.00, 10.00),
    "claude-sonnet-4-6": _price(3.00, 15.00),
    "claude-sonnet-4-5": _price(3.00, 15.00),
    # Haiku.
    "claude-haiku-5-5": _price(0.10, 0.50),
    "claude-haiku-4-5": _price(1.00, 5.00),
}

"""Anthropic for Marvin: Claude through the Messages API, as an AI provider plugin.

One entry point in ``marvin.ai_providers``: ``anthropic``. It replaces Marvin core's built-in provider of
the same slug and reads the same setting (``ANTHROPIC_API_KEY``) or a workspace's own key.
"""

from marvin_integration_sdk.ai import AIProviderPlugin

from .provider import AnthropicProvider

__all__ = ["AnthropicProvider", "plugin"]
__version__ = "0.1.0"

plugin = AIProviderPlugin(slug="anthropic", name="Anthropic", provider=AnthropicProvider, description="Claude, through Anthropic's Messages API.")

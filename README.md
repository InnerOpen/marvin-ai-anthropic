# marvin-ai-anthropic

Anthropic for [Marvin](https://github.com/InnerOpen/marvin), as an AI provider plugin: Claude through the
**Messages API**.

- Chat completions, **tool calling** (`tool_use` / `tool_result` blocks; consecutive results are merged
  into one user turn), **vision** (base64 images) and **structured output** (a forced call of one tool
  whose input schema is the output schema).
- `default_model` is `claude-sonnet-5`. No model names are coded anywhere except the price table and
  the one-token model `test_connection` uses: a new model works the day it ships.
- `list_models` lists what the account can use (`GET /v1/models`).

## Install

Marvin installs site-wide plugins at pod start from the chart's `plugins.packages` list. Only a
platform admin installs plugins; every workspace can then pick the provider in **Settings → AI**.

```yaml
plugins:
  packages:
    - https://github.com/InnerOpen/marvin-integration-sdk/archive/refs/heads/develop.tar.gz
    - https://github.com/InnerOpen/marvin-ai-anthropic/archive/refs/heads/main.tar.gz
```

Requires Marvin with the AI provider contract (`marvin-integration-sdk` 0.8+). The plugin registers in
the `marvin.ai_providers` entry-point group under the slug `anthropic`. Marvin core still has a built-in
`anthropic` provider; once this plugin is installed it **replaces** it (the log says `AI provider plugin
'anthropic' replaces core's built-in 'anthropic'`) and reads the same setting, so nothing else changes.
**Admin → Plugins** lists `marvin-ai-anthropic` with kind *AI provider*.

> pip puts `anthropic` and its dependencies (`httpx`, `pydantic`, `anyio`, `jiter`…) into the plugins
> volume, and `PYTHONPATH` comes before the image's site-packages, so those copies replace the image's
> own versions at runtime. Check that the versions pip picks match Marvin's lock before installing this
> in production.

## Settings

| Setting | |
|---|---|
| `ANTHROPIC_API_KEY` | secret, masked |

Platform credentials apply to every workspace in *platform* credential mode; a workspace in *workspace*
mode uses its own key (a workspace secret).

## Prices

`prices.py` holds Anthropic's prices per model (USD per million tokens, base input and output), taken
from [LiteLLM's model price list](https://github.com/BerriAI/litellm) on 2026-10-10. Base ids only: a
dated snapshot (`claude-haiku-4-5-20251001`) takes its base id's price; a model that isn't listed has no
estimate, and Marvin shows the run as unpriced rather than free. A new model's price is a release of this
package, not of Marvin. `suggested_models` is the same table.

## Develop

```sh
uv run --extra dev pytest -q -rs
uv run --extra dev ruff check . && uv run --extra dev ruff format --check .
```

The tests never call Anthropic. `tests/fake_server.py` is a fake Messages API behind
`httpx.MockTransport`, so requests and answers still go through the `anthropic` SDK's own encoding and
parsing; the SDK's conformance kit (`marvin_integration_sdk.ai.testing`) runs over it. `tests/test_live.py`
makes one real call when `ANTHROPIC_API_KEY` is set and is skipped otherwise.

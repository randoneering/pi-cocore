# pi-cocore

Use open-source models served through [cocore.dev](https://cocore.dev), directly inside [pi](https://pi.dev).
Providers can run on Apple Silicon or Linux.
Linux support is available on upstream's [rchowe/linux branch](https://github.com/graze-social/cocore/tree/rchowe/linux), using OpenAI-compatible backends such as llama.cpp, Ollama, or vLLM.
See that branch's [Linux provider guide](https://github.com/graze-social/cocore/blob/rchowe/linux/docs/linux-provider.md) for setup.

## Install

After the first npm release is published:

```bash
pi install npm:pi-cocore
```

Install the current fork directly from GitHub:

```bash
pi install git:github.com/randoneering/pi-cocore
```

The npm package carries the `pi-package` keyword for discovery in the [pi package gallery](https://pi.dev/packages).
Gallery indexing can lag publication.

## Setup

1. Get your API key from [cocore.dev](https://cocore.dev)
2. Start pi; on first run you'll be prompted for the key
3. Or run `/cocore-setup` at any time to configure your key, country, or minimum tray-provider version

`/cocore-setup` prompts for three inputs in sequence.
Leave an input blank to keep its current value.
Enter `-` for country or minimum provider version to clear that pin without changing the other settings.
After saving, restart pi or run `/reload` to activate the settings and refresh the model list.

| Field | Example | Effect |
|-------|---------|--------|
| API key | `cocore-…` | Required. Authenticates requests. |
| Country | `US` | Optional. ISO 3166-1 alpha-2. Routes only to providers advertising that region; this is a provider self-claim. |
| Minimum provider version | `0.9.32` | Optional. Excludes providers running an older tray release. |

## Usage

Once configured, four Co/Core routing tiers appear in the model picker (`Ctrl+P`) alongside your other providers.
Each tier shares the same model list but picks providers differently.
Endpoint paths below use the host `https://cocore.dev`:

| Picker entry | Endpoint | When to use |
|--------------|----------|-------------|
| Co/Core | `/api/v1/chat/completions` | Default. Any online provider. |
| Co/Core (private) | `/api/v1/private/chat/completions` | Routing limited to providers on your friends list. |
| Co/Core (verified) | `/api/v1/verified/chat/completions` | Only cryptographically-attested providers (hardware-attested by default). |
| Co/Core (probono) | `/api/v1/probono/chat/completions` | Only providers that opt to serve you for free. |

Supported models include:

| Model | Context |
|-------|---------|
| Qwen 2.5 (0.5B – 32B) | 32K – 128K |
| Qwen 3 / 3.5 / 3.6 | 128K |
| Gemma 3 / 4 | 32K – 128K |

Capabilities (context window, max tokens, reasoning) are automatically derived from each model's ID.

### Tool calling

All four routing tiers use the same tool-calling paths:

- Exact model/backend IDs audited in [upstream PR #196](https://github.com/graze-social/cocore/pull/196) use OpenAI `tools` and structured `tool_calls`.
  These include vetted Qwen 2.5 Instruct 0.5B/3B/7B/32B MLX, Qwen 3.5/3.6 MLX, and Ornith pairings.
  A family name alone does not qualify an unlisted backend or quantization.
- Other Gemma and Qwen IDs use model-native text instructions and response parsing.
  When tools are active, the extension buffers text until the response finishes, then emits structured tool calls before Pi's execution loop continues.
  Gemma recovery accepts unquoted argument keys and `tool:action` names only when the active tool schema declares that action.
  Unknown tools, conflicting actions, invalid arguments, and token-limited responses do not execute as recovered text calls.
- Other model families use OpenAI tools and require upstream support; they have no text-tool fallback.

The `/models` catalog includes chat-only models; appearing in the picker does not guarantee working tool calls.
Native tool support also requires a connected machine that passes its startup canary, and owners can disable it.
Even an allowlisted model can return `400 tool_calls_not_supported` when no capable provider is available.
The extension surfaces that error without automatically switching routing tiers or tool-calling paths.

### Thinking-mode handling

When a model's reasoning is set to "off" in pi's model picker (or via `/think`), the request body forwards `chat_template_kwargs: { enable_thinking: false }` so the chat template suppresses `<think>...</think>` tokens at the source.
Local serving stacks don't always honor `enable_thinking`, so leaked `<think>...</think>` ranges are also stripped from the final text block.

### Error handling

Server-side error codes map to actionable messages.
Retries keep the same routing tier and dispatch pins:

| Code | Retryable | Meaning |
|------|-----------|---------|
| `insufficient_credits` | no | Balance cannot cover this request; top up at cocore.dev/account |
| `model_not_found` | no | Refresh with `/reload` or pick another model |
| `tool_calls_not_supported` | no | No connected provider has live tool support for this model; pick another supported model or disable tools |
| `no_providers_connected` | yes | No providers are online |
| `no_providers_for_country` | yes | No providers match the country pin; enter `-` for country in `/cocore-setup` to clear it |
| `no_providers_for_version` | yes | No providers meet the version floor; enter `-` for minimum provider version in `/cocore-setup` to clear it |
| `no_verified_providers` | yes | No cryptographically verified providers qualify |
| `no_friends_available` | yes | Private tier has no online friends |
| `no_friends_for_model` | no | No friends serve this model |
| `no_pro_bono_providers` | yes | No connected provider serves you for free |
| `pro_bono_lookup_failed` | yes | The upstream pro-bono provider lookup failed |
| `onboarding_required` | no | Complete account onboarding at cocore.dev |
| `authentication_error` | no | Replace the rejected API key with `/cocore-setup` |

## Requirements

- pi (latest)
- Node.js 22.19.0 or newer
- A Co/Core API key from [cocore.dev](https://cocore.dev)

## Tests

```sh
npm ci --ignore-scripts
npm test
```

Runs six suites:

| Suite | Coverage |
|-------|----------|
| `convert-messages.test.mjs` | Text-tool and OpenAI tool history round-trips |
| `parse-tool-calls.test.mjs` | Gemma/Qwen parsers, literal-quote escapes, unquoted keys, and guarded action recovery |
| `empty-response.test.mjs` | Empty SSE responses surface errors |
| `thinking-mode.test.mjs` | Thinking controls and leaked reasoning cleanup |
| `routing-and-errors.test.mjs` | Exact allowlist, dispatch retries/errors, registration, routing pins, setup, catalog model paths across tiers, and real agent-loop execution |
| `packaging.test.mjs` | Published file boundary, host peers, and release-tag/version validation |

## Releases

1. Store a granular npm token with publish permission as the `NPM_TOKEN` repository secret.
   If the account requires 2FA, the token needs bypass permission for unattended publishing.
   Enter it through GitHub settings or `gh secret set NPM_TOKEN --repo randoneering/pi-cocore`; never commit it.
2. Update the version with `npm version <version> --no-git-tag-version`, run `npm test`, and commit the source and lockfile.
   For the first release, the existing version is `1.0.0`.
3. Push a matching `v<version>` tag and publish a GitHub release for that tag.
   The publishing workflow requires the tag version to match the committed `package.json`.
4. The workflow tests the package and publishes to npm with provenance.
   Stable versions use `latest`; prereleases use `next`.
   Missing authentication fails the workflow instead of silently skipping publication.

CI tests Node.js 22.19.0 and 24.
The npm package ships only the extension, README, package metadata, and license notices.

## License

This fork is licensed under [GPL-3.0-or-later](LICENSE).
It derives from [Will Atlas's pi-cocore](https://github.com/willnewby/pi-cocore), whose upstream manifest declares MIT.
[NOTICE](NOTICE) preserves upstream attribution and MIT permission terms.

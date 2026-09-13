# pi-cocore

Run open-source models on Apple Silicon via [cocore.dev](https://cocore.dev) — directly inside [pi](https://pi.dev).

## Install

```bash
pi install git:github.com/willnewby/pi-cocore
```

## Setup

1. Get your API key from [console.cocore.dev](https://console.cocore.dev)
2. Start pi — on first run you'll be prompted for the key
3. Or run `/cocore-setup` at any time to configure (or change) your key

## Usage

Once configured, Co/Core models appear in the model picker (`Ctrl+P`) alongside your other providers. Supported models include:

| Model | Context |
|-------|---------|
| Qwen 2.5 (0.5B – 32B) | 32K – 128K |
| Qwen 3 | 128K |
| Gemma 3 / 4 | 32K – 128K |

Capabilities (context window, max tokens, reasoning) are automatically derived from each model's ID.

### Thinking-mode handling

When a model's reasoning is set to "off" in pi's model picker (or via `/think`), the request body forwards `chat_template_kwargs: { enable_thinking: false }` so the chat template suppresses `<think>...</think>` tokens at the source. As a defensive backstop — local serving stacks don't always honor `enable_thinking` consistently — any `<think>...</think>` ranges that do leak through are stripped from the final text block before it reaches the UI.

## Requirements

- pi (latest)
- A Co/Core API key from [console.cocore.dev](https://console.cocore.dev)

## Tests

```sh
npm test
```

Runs `convert-messages.test.mjs` (tool-call/tool-result history round-trips
through text for Gemma/Qwen models and through OpenAI `tool_calls` /
`role: tool` for everything else) and `parse-tool-calls.test.mjs`
(Gemma/Qwen output envelope parsers and the literal-quote escape).

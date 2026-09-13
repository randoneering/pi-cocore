// Self-check for thinking-mode control.
//
// Run with:
//   node --experimental-strip-types --no-warnings test/thinking-mode.test.mjs
//
// Two behaviors need pinning:
//
//  1. When pi passes `reasoning: "off"` in SimpleStreamOptions, the cocore
//     request body must include `chat_template_kwargs: { enable_thinking:
//     false }` so the Qwen3 chat template suppresses thinking tokens at
//     the source. Without this, the model emits <think>...</think> inline
//     in `delta.content`.
//
//  2. Even when (1) works, models can still leak thinking content (bad
//     chat template, partial `chat_template_kwargs` support on the server,
//     or a model that ignores the flag). The text block must therefore
//     be defensively stripped of `<think>...</think>` ranges at
//     `text_end` time so the stored message is clean.
//
// Both are tested directly here; integration through `streamCocore` is
// covered by manual pi sessions.

import assert from "node:assert/strict";
import {
  stripThinkingContent,
  buildCocoreRequestBody,
} from "../extensions/cocore.ts";

// ── stripThinkingContent ────────────────────────────────────────────────────

assert.equal(
  stripThinkingContent("<think>hidden</think> visible"),
  "visible",
  "basic <think>...</think> must be stripped",
);

assert.equal(
  stripThinkingContent("<think>hidden</think> visible"),
  "visible",
  "alternate close token with leading slash must be tolerated",
);

assert.equal(
  stripThinkingContent("<think>unclosed to end of string"),
  "",
  "unclosed <think> must be stripped to end of text",
);

assert.equal(
  stripThinkingContent("no think block here"),
  "no think block here",
  "text without think blocks must be left untouched",
);

assert.equal(
  stripThinkingContent("<think>a</think><think>b</think>"),
  "",
  "multiple sequential think blocks must all be stripped",
);

assert.equal(
  stripThinkingContent("before<think>hidden</think>after"),
  "beforeafter",
  "inline think block in middle must be stripped",
);

assert.equal(
  stripThinkingContent(
    "<think>multi\nline\nhidden\nacross many lines</think> visible",
  ),
  "visible",
  "multiline think block must be stripped",
);

assert.equal(
  stripThinkingContent("visible<think>still thinking"),
  "visible",
  "unclosed think block at end must not swallow preceding content",
);

assert.equal(
  stripThinkingContent(
    "<think>The user is asking about my training data cutoff date, which is a factual question\nabout my capabilities that doesn't require tool usage or code execution.\n</think>\nI was last trained on data from October 2025.",
  ),
  "I was last trained on data from October 2025.",
  "real captured fragment from a Qwen3 session must be stripped of its thinking",
);

// ── buildCocoreRequestBody: reasoning: "off" ────────────────────────────────

const dummyModel = {
  id: "mlx-community/qwen3-4b",
  api: "openai-completions",
  provider: "cocore",
};

const dummyContext = {
  messages: [{ role: "user", content: "hi" }],
  systemPrompt: "you are a test",
};

{
  const body = buildCocoreRequestBody(dummyModel, dummyContext, "qwen", {
    reasoning: "off",
  });
  assert.ok(
    body.chat_template_kwargs,
    "reasoning: 'off' must produce chat_template_kwargs in the body",
  );
  assert.deepEqual(
    body.chat_template_kwargs,
    { enable_thinking: false },
    "enable_thinking must be false so the Qwen3 chat template suppresses thinking",
  );
}

{
  // Undefined reasoning (most common case — pi doesn't pass a level for
  // models that don't expose thinking controls) must NOT inject the flag,
  // otherwise we'd override the server's default.
  const body = buildCocoreRequestBody(dummyModel, dummyContext, "qwen", {});
  assert.equal(
    body.chat_template_kwargs,
    undefined,
    "absent reasoning must not set chat_template_kwargs",
  );
}

{
  // Non-off reasoning levels are NOT standardized across MLX/Qwen serving
  // stacks, so we deliberately don't map them. The model decides; our
  // defensive `stripThinkingContent` covers whatever the server emits.
  const body = buildCocoreRequestBody(dummyModel, dummyContext, "qwen", {
    reasoning: "low",
  });
  assert.equal(
    body.chat_template_kwargs,
    undefined,
    "non-off reasoning levels must not set chat_template_kwargs (no standard mapping)",
  );
}

{
  // reasoning: "off" must apply regardless of model family — Gemma 3/4
  // may also honor the same kwarg name.
  const body = buildCocoreRequestBody(dummyModel, dummyContext, "gemma", {
    reasoning: "off",
  });
  assert.deepEqual(
    body.chat_template_kwargs,
    { enable_thinking: false },
    "reasoning: 'off' must produce chat_template_kwargs for gemma too",
  );
}

console.log("ok - thinking-mode self-check passed");

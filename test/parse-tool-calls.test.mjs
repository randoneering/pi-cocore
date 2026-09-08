// Self-check for the tool-call output parsers.
//
// Run with:
//   NODE_PATH=/path/to/pi-monorepo/node_modules \
//     node --experimental-strip-types --no-warnings \
//     test/parse-tool-calls.test.mjs
//
// Regression coverage for the gemma/qwen output side. PR #1 fixed the
// *input* path (history collapse); this covers the *output* path
// (parsing model-native tool calls back into structured blocks) —
// specifically the variants the model actually emits through a local
// serving stack with a Gemma/Qwen chat template.

import assert from "node:assert/strict";
import {
  parseToolCalls,
  parseGemma4ToolCalls,
  parseGemma3ToolCalls,
  fixCocoreToolCalls,
  normalizeModelText,
} from "../extensions/cocore.ts";

// --- normalizeModelText ----------------------------------------------------
{
  // `<|"|>` is the model's literal-quote escape inside a JSON value;
  // mapping it to `\"` keeps the surrounding JSON parseable when the
  // value itself contains `"` chars (e.g. a bash command like
  // `--pretty=format:"%h %s"`).
  assert.equal(
    normalizeModelText('a<|"|>b'),
    'a\\"b',
    "literal-quote escape must become an escaped quote",
  );
  assert.equal(
    normalizeModelText('plain text without escapes'),
    'plain text without escapes',
    "no-op when no escapes are present",
  );
}

// --- regression: real broken fragment from pi session ----------------------
//
// Captured verbatim from a `google/gemma-4-12b` run via pi-cocore on
// 2026-09-08: the model emitted `<|tool_call|>` (close-marker) as the
// opener, leaked `<|"|>` literal-quote escapes through the tokenizer,
// and mangled the close to `<tool_call|>` (missing leading pipe). The
// whole call silently dropped before the fix.
const BROKEN_BASH_FRAGMENT =
  `<|tool_call|>call:bash{command:<|"|>git fetch origin && git log -n 1 --pretty=format:"%h %s" ` +
  `origin/main -- #{44} 2>/dev/null || git log -n 1 --pretty=format:"%h %s" origin/master -- #{44} ` +
  `2>/dev/null || git log -n 1 --pretty=format:"%h %s" main -- #{44} 2>/dev/null || git log -n 1 ` +
  `--pretty=format:"%h %s" master -- #{44} 2>/dev/null<|"|>}<tool_call|>`;

{
  const out = parseToolCalls(BROKEN_BASH_FRAGMENT, "gemma");
  assert.equal(out.length, 1, "broken fragment must yield exactly one match");
  assert.equal(out[0].name, "bash");
  assert.ok(
    typeof out[0].arguments.command === "string",
    "arguments.command must survive as a string",
  );
  assert.ok(
    out[0].arguments.command.startsWith("git fetch origin"),
    "command must begin with the git fetch chain",
  );
  assert.ok(
    out[0].arguments.command.includes("--pretty=format:"),
    "nested quotes in the inner format string must survive normalization",
  );
  assert.ok(
    out[0].arguments.command.endsWith("2>/dev/null"),
    "command must end with the final 2>/dev/null fallback",
  );
}

// --- nominal Gemma 4 format still parses ----------------------------------
{
  const wellFormed = `<|tool_call>call:nixos_search{"query":"neovim"}<|tool_call|>`;
  const out = parseToolCalls(wellFormed, "gemma");
  assert.equal(out.length, 1);
  assert.equal(out[0].name, "nixos_search");
  assert.deepEqual(out[0].arguments, { query: "neovim" });
}

// --- swapped delimiters: opener used as closer, etc. -----------------------
{
  // Model used the close-marker as opener; closer is missing the leading pipe.
  const swapped = `<|tool_call|>call:bash{"cmd":"ls"}<tool_call|>`;
  const out = parseToolCalls(swapped, "gemma");
  assert.equal(out.length, 1, "swapped delimiters must still parse");
  assert.equal(out[0].name, "bash");
  assert.deepEqual(out[0].arguments, { cmd: "ls" });
}

// --- multiple calls in one stream ------------------------------------------
{
  // Gemma 4 format: each call has its own opener/closer pair.
  const two =
    `<|tool_call>call:a{"x":1}<|tool_call|>` +
    `<|tool_call>call:b{"y":2}<|tool_call|>`;
  const out = parseToolCalls(two, "gemma");
  assert.equal(out.length, 2);
  assert.equal(out[0].name, "a");
  assert.deepEqual(out[0].arguments, { x: 1 });
  assert.equal(out[1].name, "b");
  assert.deepEqual(out[1].arguments, { y: 2 });
}

// --- nested object inside arguments survives the lazy quantifier ----------
{
  const nested = `<|tool_call>call:wrap{"outer":{"inner":42}}<|tool_call|>`;
  const out = parseToolCalls(nested, "gemma");
  assert.equal(out.length, 1);
  assert.equal(out[0].name, "wrap");
  assert.deepEqual(out[0].arguments, { outer: { inner: 42 } });
}

// --- direct parseGemma4ToolCalls sees normalized input --------------------
// (Sub-parsers are exported but assume normalized text. Callers that
// bypass parseToolCalls should call normalizeModelText themselves.)
{
  const envelope = `<|tool_call|>call:bash{command:<|"|>ls<|"|>}<|tool_call|>`;
  const normalized = normalizeModelText(envelope);
  const out = parseGemma4ToolCalls(normalized);
  assert.equal(out.length, 1);
  assert.equal(out[0].name, "bash");
  assert.equal(out[0].arguments.command, "ls");
}

// --- Gemma 3 fallback still works -----------------------------------------
{
  const g3 = `<|tool_call|>nixos_search{"query":"neovim"}<|tool_call|>`;
  const out = parseGemma3ToolCalls(g3);
  assert.equal(out.length, 1);
  assert.equal(out[0].name, "nixos_search");
  assert.deepEqual(out[0].arguments, { query: "neovim" });
}

// --- qwen path untouched --------------------------------------------------
{
  const qwen = `<tool_call>
{"name":"nixos_search","arguments":{"query":"neovim"}}
</tool_call>`;
  const out = parseToolCalls(qwen, "qwen");
  assert.equal(out.length, 1);
  assert.equal(out[0].name, "nixos_search");
  assert.deepEqual(out[0].arguments, { query: "neovim" });
}

// --- fixCocoreToolCalls integration: broken fragment becomes a toolCall ---
{
  const assistant = {
    role: "assistant",
    content: [{ type: "text", text: BROKEN_BASH_FRAGMENT }],
  };
  const out = fixCocoreToolCalls(assistant, "gemma");
  assert.equal(out.role, "assistant");
  const toolCallBlocks = out.content.filter((b) => b.type === "toolCall");
  assert.equal(toolCallBlocks.length, 1, "fixer must emit one toolCall block");
  assert.equal(toolCallBlocks[0].name, "bash");
  assert.ok(
    toolCallBlocks[0].arguments.command.startsWith("git fetch origin"),
    "fixer must preserve the full command chain in arguments",
  );
}

console.log("ok - parse-tool-calls self-check passed");

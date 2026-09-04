// Self-check for convertMessagesForOpenAI.
//
// Run with:
//   NODE_PATH=/path/to/pi-monorepo/node_modules \
//     node --experimental-strip-types --experimental-detect-module \
//     test/convert-messages.test.mjs
//
// Asserts the gemma/qwen branch collapses toolCall + toolResult history
// into text, and the non-family branch keeps OpenAI tool_calls / role:tool.

import assert from "node:assert/strict";
import { convertMessagesForOpenAI, getModelFamily } from "../extensions/cocore.ts";

const toolCallBlock = (name, args, id = "tc-1") => ({
  type: "toolCall",
  id,
  name,
  arguments: args,
});

const toolResult = (text, id = "tc-1", name = "mcp__kaneo") => ({
  role: "toolResult",
  toolCallId: id,
  toolName: name,
  content: [{ type: "text", text }],
});

const assistantWithToolCall = (text, name, args) => ({
  role: "assistant",
  content: [
    { type: "text", text },
    toolCallBlock(name, args),
  ],
});

// --- gemma branch: tool history collapses to text -----------------------
{
  const msgs = [
    { role: "user", content: "list my tasks" },
    assistantWithToolCall("Let me check.", "mcp__kaneo", { project: "forge" }),
    toolResult("Found 3 tasks.", "tc-1", "mcp__kaneo"),
  ];
  const out = convertMessagesForOpenAI(msgs, "gemma");

  // assistant content must be plain text (no tool_calls field), and must
  // contain a gemma-style tool_call segment.
  assert.equal(out.length, 3, "expected 3 messages");
  assert.equal(out[1].role, "assistant");
  assert.equal(out[1].tool_calls, undefined, "gemma path must not emit tool_calls");
  assert.ok(
    typeof out[1].content === "string" && out[1].content.includes("<|tool_call|>mcp__kaneo"),
    "gemma path must inline <|tool_call|>name{...}",
  );
  assert.ok(
    out[1].content.includes("Let me check."),
    "prior text content preserved alongside tool call",
  );

  // toolResult becomes a user-role message describing the function output.
  assert.equal(out[2].role, "user");
  assert.ok(out[2].content.includes("mcp__kaneo"), "toolName surfaced to model");
  assert.ok(out[2].content.includes("Found 3 tasks."), "tool result text carried over");
  assert.ok(!("tool_call_id" in out[2]), "toolResult in gemma path must not be role:tool");
}

// --- qwen branch: same collapse, qwen-style tag --------------------------
{
  const msgs = [
    assistantWithToolCall("", "nixos_search", { query: "neovim" }),
    toolResult("10 results.", "tc-2", "nixos_search"),
  ];
  const out = convertMessagesForOpenAI(msgs, "qwen");
  assert.equal(out[0].tool_calls, undefined, "qwen path must not emit tool_calls on assistant");
  assert.ok(
    out[0].content.includes('"name":"nixos_search"'),
    "qwen path must inline tool-call JSON with name + arguments",
  );
  assert.ok(out[0].content.includes("arguments"), "qwen path must include arguments key");
  assert.equal(out[1].role, "user");
  assert.ok(out[1].content.includes("10 results."));
}

// --- non-family (openai native) branch unchanged -------------------------
{
  const msgs = [
    assistantWithToolCall("hi", "bash", { cmd: "ls" }),
    toolResult("file.txt", "tc-3", "bash"),
  ];
  const out = convertMessagesForOpenAI(msgs); // no family
  assert.equal(out[0].role, "assistant");
  assert.ok(Array.isArray(out[0].tool_calls), "non-family path keeps tool_calls");
  assert.equal(out[0].tool_calls[0].function.name, "bash");
  assert.equal(out[1].role, "tool");
  assert.equal(out[1].tool_call_id, "tc-3");
}

// --- getModelFamily sanity ---------------------------------------------
assert.equal(getModelFamily("qwen3-7b"), "qwen");
assert.equal(getModelFamily("google/gemma-4-12b"), "gemma");
assert.equal(getModelFamily("llama-3.1-8b"), null);

console.log("ok - convert-messages self-check passed");

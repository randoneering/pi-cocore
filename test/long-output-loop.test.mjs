// Self-check for the post-fix long-output regression.
//
// Run with:
//   node --experimental-strip-types --no-warnings test/long-output-loop.test.mjs
//
// Reproduces the f17b420d failure: gemma-4 emits a real `write` tool call
// to deliver the artifact, then follows up with a turn whose text buffer
// is just a literal `<|tool_call|>write{...}` envelope and whose
// stopReason is "stop". The previous 2026-10-02 fix made `looksTruncated`
// flag the empty / envelope-only text as truncated and `runCompletionRetry`
// issued a continuation request that the model had nothing left to say
// to. The supervisor saw an errorMessage and the acceptance parser saw no
// fenced report.
//
// The fix must:
//
//   1. Distinguish "empty final text-only turn after a successful tool
//      call" from "truncated report" in `looksTruncated`.
//   2. Skip completion-retry when the assistant's most recent turn
//      successfully wrote the artifact through a real tool call, or
//      when the buffered text is empty / just a literal `<|tool_call|>`
//      envelope after a `write`/`edit` tool call.
//   3. Treat literal `<|tool_call|>...` text as a soft signal to check
//      tool-call parsing rather than as truncation when the model also
//      has no pending content.
//   4. Preserve real truncation detection: unterminated JSON, mid-word,
//      unbalanced braces must still be reported as truncated.

import assert from "node:assert/strict";
import { test } from "node:test";
import { looksTruncated, streamCocore } from "../extensions/cocore.ts";

function sseResponse(chunks) {
  const data = chunks.map((chunk) => `data: ${JSON.stringify(chunk)}\n\n`).join("");
  return new Response(data + "data: [DONE]\n\n", {
    headers: { "Content-Type": "text/event-stream" },
  });
}

async function collectEvents(stream) {
  const events = [];
  for await (const event of stream) events.push(event);
  return events;
}

const gemmaModel = {
  id: "google/gemma-4-12b",
  api: "openai-completions",
  provider: "cocore",
  baseUrl: "https://test.invalid/api/v1",
};

const readDecl = {
  name: "read",
  description: "Read a file",
  parameters: { type: "object", properties: { path: { type: "string" } }, required: ["path"] },
};
const writeDecl = {
  name: "write",
  description: "Write a file",
  parameters: {
    type: "object",
    properties: { path: { type: "string" }, content: { type: "string" } },
    required: ["path", "content"],
  },
};
const editDecl = {
  name: "edit",
  description: "Edit a file",
  parameters: { type: "object", properties: { path: { type: "string" } }, required: ["path"] },
};

// ── looksTruncated soft exclusions ─────────────────────────────────────────

const writeEnvelope = '<|tool_call|>write{"path":"/tmp/x.md","content":"# hi"}';
const longWriteEnvelope =
  '<|tool_call|>write{"path":"/tmp/postfix-long-output.md","content":"# 1. section\\n\\nLONG-OUTPUT-PROBE-COMPLETE\\n\\n```acceptance-report\\n{\\n  \\"criteriaSatisfied\\": [{\\"id\\": \\"complete-probe\\", \\"status\\": \\"satisfied\\"}]\\n}\\n```"}';

await test("looksTruncated returns false for empty text after a successful write call", () => {
  const message = {
    role: "assistant",
    content: [
      { type: "toolCall", id: "tc-1", name: "write", arguments: { path: "/tmp/x.md", content: "# hi" } },
      { type: "text", text: "" },
    ],
  };
  assert.equal(looksTruncated(message), false, "empty text after a write must not be flagged");
});

await test("looksTruncated returns false for a literal <|tool_call|> envelope after a write", () => {
  const message = {
    role: "assistant",
    content: [
      { type: "toolCall", id: "tc-1", name: "write", arguments: { path: "/tmp/x.md", content: "# hi" } },
      { type: "text", text: longWriteEnvelope },
    ],
  };
  assert.equal(
    looksTruncated(message),
    false,
    "literal <|tool_call|> envelope after a write must not be flagged",
  );
});

await test("looksTruncated returns false for an unterminated <|tool_call|> envelope after a write", () => {
  const message = {
    role: "assistant",
    content: [
      { type: "toolCall", id: "tc-1", name: "write", arguments: { path: "/tmp/x.md", content: "# hi" } },
      { type: "text", text: "<|tool_call|>" },
    ],
  };
  assert.equal(
    looksTruncated(message),
    false,
    "bare <|tool_call|> opener after a write must not be flagged",
  );
});

await test("looksTruncated returns false for a literal <|tool_call|> envelope after an edit call", () => {
  const message = {
    role: "assistant",
    content: [
      { type: "toolCall", id: "tc-1", name: "edit", arguments: { path: "/tmp/x.md" } },
      { type: "text", text: longWriteEnvelope },
    ],
  };
  assert.equal(looksTruncated(message), false);
});

await test("looksTruncated returns true for an unterminated JSON string (no write call)", () => {
  const message = {
    role: "assistant",
    content: [{ type: "text", text: '{"criteriaSatisfied":[{"id":"x","status":"' }],
  };
  assert.equal(looksTruncated(message), true, "unterminated JSON must still be flagged");
});

await test("looksTruncated returns true for mid-word truncation (no write call)", () => {
  const message = {
    role: "assistant",
    content: [{ type: "text", text: "This minimizes the risk of halluc" }],
  };
  assert.equal(looksTruncated(message), true, "mid-word truncation must still be flagged");
});

await test("looksTruncated returns true for unbalanced braces (no write call)", () => {
  const message = {
    role: "assistant",
    content: [{ type: "text", text: "# Section 1. body. {\"a\":1" }],
  };
  assert.equal(looksTruncated(message), true, "unbalanced braces must still be flagged");
});

await test("looksTruncated returns false for a literal <|tool_call|> envelope with no other content", () => {
  // A buffered text that is just a literal <|tool_call|>... envelope is
  // the model's text-shaped attempt at a tool call. The continuation
  // would not produce new prose. Only apply when there is no other
  // pending content in the buffer; mixed text + envelope still falls
  // through to the structural checks.
  const message = {
    role: "assistant",
    content: [{ type: "text", text: longWriteEnvelope }],
  };
  assert.equal(looksTruncated(message), false);
});

await test("looksTruncated returns false for empty text with no tool calls", () => {
  const message = { role: "assistant", content: [{ type: "text", text: "" }] };
  assert.equal(looksTruncated(message), false);
});

await test("looksTruncated returns false for a complete report with a terminal marker", () => {
  const message = {
    role: "assistant",
    content: [{ type: "text", text: "LONG-OUTPUT-PROBE-COMPLETE" }],
  };
  assert.equal(looksTruncated(message), false, "completion marker is a sentence-final pattern");
});

await test("looksTruncated returns false for a complete acceptance-report JSON fence", () => {
  const message = {
    role: "assistant",
    content: [{
      type: "text",
      text: '```acceptance-report\n{"criteriaSatisfied":[{"id":"x","status":"satisfied"}]}\n```',
    }],
  };
  assert.equal(looksTruncated(message), false);
});

// ── streamCocore integration: empty text after a successful tool call ─────

// The harness executes the write tool call that streams out of the model.
// The follow-up text is just a literal <|tool_call|>write{...} envelope.
// The previous fix would mark this as truncated and burn the retry budget.
// The new code must let it through with stopReason=toolUse and no
// errorMessage.
await test("streamCocore does not surface truncation when a write call already landed and text repeats the envelope", async () => {
  const baseContext = {
    messages: [{ role: "user", content: "read fixtures and write a 16-section report" }],
    systemPrompt: "",
    tools: [readDecl, writeDecl, editDecl],
  };
  let requests = 0;
  const events = await collectEvents(streamCocore(
    gemmaModel, baseContext, { maxRetries: 0 },
    { fetchImpl: async () => {
      requests++;
      return sseResponse([{ choices: [{
        delta: {
          content: longWriteEnvelope,
          tool_calls: [{ index: 0, id: "tc-write", type: "function", function: {
            name: "write", arguments: '{"path":"/tmp/postfix-long-output.md","content":"# 1. section"}',
          } }],
        },
        finish_reason: "tool_calls",
      }] }]);
    } },
  ));
  const last = events.at(-1);
  assert.equal(last.type, "done");
  // The text→toolCall fixer converts the envelope into a tool call, so
  // the final state has a tool call and the literal envelope text is
  // gone. The completion-retry must not fire on top of this.
  assert.equal(requests, 1, "no completion retry when the write call already landed");
  assert.equal(last.reason, "toolUse");
  assert.equal(
    last.message.errorMessage, undefined,
    `errorMessage must be unset; got: ${last.message.errorMessage}`,
  );
});

// Regression guard: real truncation must still be retried. Without a
// write tool call landing, a mid-JSON response still triggers the
// completion-retry path.
await test("streamCocore still issues a completion retry for mid-word truncation", async () => {
  const baseContext = {
    messages: [{ role: "user", content: "write a report" }],
    systemPrompt: "",
    tools: [writeDecl],
  };
  let requests = 0;
  const events = await collectEvents(streamCocore(
    gemmaModel, baseContext, { maxRetries: 0 },
    { fetchImpl: async () => {
      requests++;
      return sseResponse([{ choices: [{
        delta: { content: "This minimizes the risk of halluc" },
        finish_reason: "stop",
      }] }]);
    } },
  ));
  assert.equal(requests, 2, "mid-word truncation still gets one completion retry");
  const last = events.at(-1);
  assert.match(last.message.errorMessage ?? "", /truncat/);
});

// A bare <|tool_call|>write{...} envelope in the text buffer (no native
// tool call, no prior turn context) is the model's text-shaped attempt
// at a tool call. The completion-retry would not produce anything new;
// the soft exclusion in looksTruncated must let it pass.
await test("streamCocore does not issue a completion retry when text is just a literal envelope", async () => {
  const baseContext = {
    messages: [{ role: "user", content: "write a report" }],
    systemPrompt: "",
    tools: [writeDecl],
  };
  let requests = 0;
  const events = await collectEvents(streamCocore(
    gemmaModel, baseContext, { maxRetries: 0 },
    { fetchImpl: async () => {
      requests++;
      return sseResponse([{ choices: [{
        delta: { content: longWriteEnvelope },
        finish_reason: "stop",
      }] }]);
    } },
  ));
  assert.equal(requests, 1, "no completion retry for a bare <|tool_call|> envelope");
  const last = events.at(-1);
  assert.equal(last.type, "done");
  assert.equal(last.reason, "toolUse", "text→toolCall fixer should convert the envelope");
  const calls = last.message.content.filter((b) => b.type === "toolCall");
  assert.equal(calls.length, 1);
  assert.equal(calls[0].name, "write");
  assert.equal(
    last.message.errorMessage, undefined,
    "envelope-only turn must not surface a truncation error",
  );
});

console.log("ok - long-output-loop self-check passed");

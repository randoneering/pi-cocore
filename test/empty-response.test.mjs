// Self-check for empty-response error surfacing.
//
// Run with:
//   node --experimental-strip-types --no-warnings test/empty-response.test.mjs
//
// Drives streamCocore with an injected mock fetch that returns 200 OK but
// an empty SSE body. The current code falls through to a `done` event with
// an empty message — the user sees retries in logs and then silence. The
// expected behavior: after retries are exhausted (or with maxRetries: 0)
// with no content, the stream must emit a structured `error` event so the
// pi UI surfaces a clear failure to the user.

import assert from "node:assert/strict";
import { ReadableStream } from "node:stream/web";
import { streamCocore } from "../extensions/cocore.ts";

// ── Test fixtures ───────────────────────────────────────────────────────────

/**
 * Build a fake fetch response that returns 200 OK but an empty SSE body.
 * The reader returns `done: true` immediately, so streamCocore accumulates
 * no text and no usage — the exact "empty response" case.
 */
function emptySSEResponse() {
  return {
    ok: true,
    status: 200,
    statusText: "OK",
    headers: new Headers(),
    body: new ReadableStream({
      start(controller) {
        controller.close();
      },
    }),
    text: async () => "",
  };
}

/** Drain the stream into a plain array of events. */
async function collectEvents(stream) {
  const events = [];
  for await (const event of stream) events.push(event);
  return events;
}

const dummyModel = {
  id: "mlx-community/qwen3-test",
  api: "openai-completions",
  provider: "cocore",
  baseUrl: "http://test.invalid",
};

const dummyContext = {
  messages: [{ role: "user", content: "ping" }],
  systemPrompt: "you are a test",
};

// ── Test: empty response surfaces as error, not silent done ─────────────────

{
  const mockFetch = async () => emptySSEResponse();

  const events = await collectEvents(
    streamCocore(
      dummyModel,
      dummyContext,
      { maxRetries: 0, apiKey: "test-key" },
      { fetchImpl: mockFetch },
    ),
  );

  const startEvent = events.find((e) => e.type === "start");
  assert.ok(startEvent, "expected a start event before any error");

  const errorEvent = events.find((e) => e.type === "error");
  assert.ok(
    errorEvent,
    "expected an error event when the response body is empty (currently a silent `done` slips through)",
  );
  assert.equal(errorEvent.reason, "error", "error reason must be 'error'");
  assert.ok(
    typeof errorEvent.error?.errorMessage === "string" &&
      errorEvent.error.errorMessage.length > 0,
    "error event must carry a non-empty errorMessage so the user can see what failed",
  );
  assert.match(
    String(errorEvent.error.errorMessage),
    /empty/i,
    "errorMessage should mention the empty response so users can diagnose",
  );

  const doneEvent = events.find((e) => e.type === "done");
  assert.equal(
    doneEvent,
    undefined,
    "must NOT emit a done event for an empty response — that path hides the failure from the UI",
  );
}

// ── Test: error event has error stopReason so callers can branch ───────────

{
  const mockFetch = async () => emptySSEResponse();

  const events = await collectEvents(
    streamCocore(
      dummyModel,
      dummyContext,
      { maxRetries: 0, apiKey: "test-key" },
      { fetchImpl: mockFetch },
    ),
  );

  const errorEvent = events.find((e) => e.type === "error");
  assert.ok(errorEvent, "expected error event");
  assert.equal(
    errorEvent.error?.stopReason,
    "error",
    "stopReason must be 'error' so downstream consumers can detect the failure",
  );
}

console.log("ok - empty-response self-check passed");

// Self-check for the upstream-API surface introduced for cocore.dev
// routing tiers, dispatch-error mapping, and the verified-tool
// allowlist (PR #196). Run with:
//
//   node --experimental-strip-types --no-warnings \
//     test/routing-and-errors.test.mjs

import assert from "node:assert/strict";
import { test } from "node:test";
import { spawnSync } from "node:child_process";
import { fileURLToPath, pathToFileURL } from "node:url";
import { createRequire } from "node:module";
import { Type } from "typebox";
import { getCurrentSystemPrompt, getCurrentTools } from "@earendil-works/pi-ai";
import {
  buildCocoreRequestBody,
  streamCocore,
  getModelFamily,
  isVerifiedToolModel,
  parseServerError,
  extractErrorCode,
  friendlyMessageFor,
  chatCompletionsPath,
  providerKey,
  providerName,
  ROUTINGS,
} from "../extensions/cocore.ts";

// ── Verified-tool allowlist ────────────────────────────────────────────────

assert.equal(
  isVerifiedToolModel("mlx-community/Qwen3.5-4B-MLX-4bit"),
  true,
  "Qwen3.5 MLX is on the verified-tool allowlist",
);
assert.equal(
  isVerifiedToolModel("mlx-community/Qwen3.5-9B-MLX-4bit"),
  true,
  "Qwen3.5 MLX 9B is on the verified-tool allowlist",
);
assert.equal(
  isVerifiedToolModel("mlx-community/Qwen3.5-0.8B-MLX-4bit"),
  true,
  "Qwen3.5 MLX 0.8B is on the verified-tool allowlist",
);
assert.equal(
  isVerifiedToolModel("mlx-community/Qwen2.5-7B-Instruct-4bit"),
  true,
  "Qwen2.5 Instruct 7B MLX is on the verified-tool allowlist",
);
assert.equal(
  isVerifiedToolModel("mlx-community/Qwen2.5-32B-Instruct-4bit"),
  true,
  "Qwen2.5 Instruct 32B MLX is on the verified-tool allowlist",
);
assert.equal(
  isVerifiedToolModel("mlx-community/Qwen2.5-0.5B-Instruct-4bit"),
  true,
  "Qwen2.5 Instruct 0.5B is on the verified-tool allowlist (PR #196 explicitly enables 0.5B/3B/7B/32B)",
);
assert.equal(
  isVerifiedToolModel("mlx-community/Qwen2.5-3B-Instruct-4bit"),
  true,
  "Qwen2.5 Instruct 3B is on the verified-tool allowlist",
);

assert.equal(
  isVerifiedToolModel("google/gemma-4-12b"),
  false,
  "Gemma 4 stays on the text-tool path",
);
assert.equal(
  isVerifiedToolModel("mlx-community/gemma-4-12B-it-8bit"),
  false,
  "Gemma 4 8-bit stays on the text-tool path",
);
assert.equal(
  isVerifiedToolModel("llama-3.1-8b"),
  false,
  "Llama is not on the verified-tool allowlist",
);
assert.equal(
  isVerifiedToolModel("mlx-community/Qwen2.5-7B"),
  false,
  "non-Instruct Qwen2.5 has no verified-tool pairing",
);
assert.equal(
  isVerifiedToolModel("mlx-community/Llama-3.3-70B-Instruct-4bit"),
  false,
  "Llama 3.3 70B Instruct is explicitly excluded by PR #196 pending a preserving parser",
);

// ── getModelFamily bypasses verified-tool models ───────────────────────────

assert.equal(
  getModelFamily("mlx-community/Qwen3.5-4B-MLX-4bit"),
  null,
  "verified-tool Qwen bypasses the text-tool family detection",
);
assert.equal(
  getModelFamily("mlx-community/Qwen2.5-7B-Instruct-4bit"),
  null,
  "verified-tool Qwen2.5 Instruct bypasses the text-tool family detection",
);
assert.equal(
  getModelFamily("google/gemma-4-12b"),
  "gemma",
  "gemma-4 stays on the text-tool path",
);
assert.equal(
  getModelFamily("qwen3-7b"),
  "qwen",
  "non-verified Qwen3 still resolves to the qwen family",
);

// ── Routing paths ──────────────────────────────────────────────────────────

assert.equal(chatCompletionsPath("open"), "/chat/completions");
assert.equal(chatCompletionsPath("private"), "/private/chat/completions");
assert.equal(chatCompletionsPath("verified"), "/verified/chat/completions");
assert.equal(chatCompletionsPath("probono"), "/probono/chat/completions");

assert.equal(providerKey("open"), "cocore");
assert.equal(providerKey("private"), "cocore-private");
assert.equal(providerKey("verified"), "cocore-verified");
assert.equal(providerKey("probono"), "cocore-probono");

assert.equal(providerName("open"), "Co/Core");
assert.equal(providerName("private"), "Co/Core (private)");
assert.equal(providerName("verified"), "Co/Core (verified)");
assert.equal(providerName("probono"), "Co/Core (probono)");

assert.deepEqual([...ROUTINGS], ["open", "private", "verified", "probono"]);

// ── Error code extraction ──────────────────────────────────────────────────

assert.equal(
  extractErrorCode('{"error":{"code":"model_not_found","message":"..."}}'),
  "model_not_found",
);
assert.equal(
  extractErrorCode('{"error":{"code":"insufficient_credits","message":"..."}}'),
  "insufficient_credits",
);
assert.equal(extractErrorCode("not json"), null, "non-JSON yields null code");
assert.equal(extractErrorCode("{}"), null, "missing error.code yields null");

// ── parseServerError retry classification ──────────────────────────────────

{
  const r = parseServerError(503, JSON.stringify({ error: { code: "no_providers_connected", message: "x" } }));
  assert.equal(r.code, "no_providers_connected");
  assert.equal(r.retryable, true, "no_providers_connected must retry");
}

{
  const r = parseServerError(503, JSON.stringify({ error: { code: "no_providers_for_country", message: "x" } }));
  assert.equal(r.retryable, true, "no_providers_for_country must retry");
}

{
  const r = parseServerError(503, JSON.stringify({ error: { code: "no_providers_for_version", message: "x" } }));
  assert.equal(r.retryable, true, "no_providers_for_version must retry");
}

{
  const r = parseServerError(503, JSON.stringify({ error: { code: "no_friends_available", message: "x" } }));
  assert.equal(r.retryable, true, "no_friends_available must retry");
}

{
  const r = parseServerError(404, JSON.stringify({ error: { code: "model_not_found", message: "x" } }), "mlx-community/Qwen3.5-4B-MLX-4bit");
  assert.equal(r.code, "model_not_found");
  assert.equal(r.retryable, false, "model_not_found must not retry");
  assert.match(
    r.friendly,
    /mlx-community\/Qwen3\.5-4B-MLX-4bit/,
    "model_not_found friendly message must include the model id",
  );
}

{
  const r = parseServerError(402, JSON.stringify({ error: { code: "insufficient_credits", message: "balance 12" } }));
  assert.equal(r.retryable, false, "insufficient_credits must not retry");
  assert.match(r.friendly, /top up/i);
}

{
  const r = parseServerError(400, JSON.stringify({ error: { code: "tool_calls_not_supported", message: "x" } }));
  assert.equal(r.retryable, false, "tool_calls_not_supported must not retry");
  assert.match(r.friendly, /tool calls/i);
}

{
  const r = parseServerError(401, JSON.stringify({ error: { code: "authentication_error", message: "x" } }));
  assert.equal(r.retryable, false, "auth errors must not retry");
  assert.match(r.friendly, /API key rejected/i);
}

{
  // Non-JSON body must still classify via HTTP status.
  const r = parseServerError(503, "Service Unavailable");
  assert.equal(r.code, null);
  assert.equal(r.retryable, true, "503 must retry when no dispatch code is present");
}

{
  // Unknown body / unknown code falls back to status heuristic.
  const r = parseServerError(429, JSON.stringify({ error: { code: "weird_new_code", message: "x" } }));
  assert.equal(r.code, "weird_new_code");
  assert.equal(r.retryable, true, "unknown dispatch code on a 429 must retry via status heuristic");
}

// ── friendlyMessageFor direct coverage ─────────────────────────────────────

assert.match(friendlyMessageFor("onboarding_required", 401, "x"), /onboard/i);
assert.match(friendlyMessageFor("no_friends_for_model", 404, "x"), /friends/i);
assert.match(friendlyMessageFor("no_pro_bono_providers", 503, "x"), /for free/i);
assert.match(friendlyMessageFor("pro_bono_lookup_failed", 502, "x"), /retry/i);
assert.match(friendlyMessageFor("no_providers_for_country", 503, "x"), /country/i);
assert.match(friendlyMessageFor("no_providers_for_version", 503, "x"), /version/i);
assert.match(friendlyMessageFor(null, 500, "boom"), /co\/core 500/);

// ── buildCocoreRequestBody routing fields ──────────────────────────────────

const dummyModel = {
  id: "mlx-community/Qwen3.5-4B-MLX-4bit",
  api: "openai-completions",
  provider: "cocore",
};

const dummyContext = {
  messages: [{ role: "user", content: "hi" }],
  systemPrompt: "you are a test",
};

{
  // Open routing, no country, no min_version — base body shape.
  const body = buildCocoreRequestBody(dummyModel, dummyContext, null, {}, { routing: "open" });
  assert.equal(body.country, undefined, "country must not appear when unset");
  assert.equal(body.min_provider_version, undefined, "min_provider_version must not appear when unset");
  assert.equal(body.min_trust, undefined, "min_trust only set on verified tier");
}

{
  // Country + min_provider_version flow through verbatim.
  const body = buildCocoreRequestBody(
    dummyModel,
    dummyContext,
    null,
    {},
    { routing: "open", country: "US", minProviderVersion: "0.9.32" },
  );
  assert.equal(body.country, "US");
  assert.equal(body.min_provider_version, "0.9.32");
}

{
  // Leading "v" on min_provider_version is stripped to match server parser.
  const body = buildCocoreRequestBody(
    dummyModel,
    dummyContext,
    null,
    {},
    { routing: "open", minProviderVersion: "v0.9.32" },
  );
  assert.equal(body.min_provider_version, "0.9.32", "leading v must be stripped");
}

{
  // Verified routing implicitly sends min_trust: hardware-attested.
  const body = buildCocoreRequestBody(
    dummyModel,
    dummyContext,
    null,
    {},
    { routing: "verified" },
  );
  assert.equal(body.min_trust, "hardware-attested", "verified tier must pin hardware-attested by default");
}

{
  // Private / probono tiers must NOT set min_trust — that field is verified-only.
  for (const routing of ["open", "private", "probono"]) {
    const body = buildCocoreRequestBody(dummyModel, dummyContext, null, {}, { routing });
    assert.equal(
      body.min_trust,
      undefined,
      `min_trust must not appear on ${routing} tier`,
    );
  }
}

{
  // Verified-tool family (null) means the request keeps OpenAI tool_calls.
  const toolsCtx = {
    messages: [{ role: "user", content: "hi" }],
    systemPrompt: "",
    tools: [{ name: "bash", description: "x", parameters: { type: "object" } }],
  };
  const body = buildCocoreRequestBody(
    dummyModel,
    toolsCtx,
    null, // family = null because verified-tool bypasses text path
    {},
    { routing: "open" },
  );
  assert.ok(Array.isArray(body.tools), "verified-tool models must keep OpenAI tools array");
  assert.equal(body.tools[0].function.name, "bash");
}

{
  // Text-tool family (gemma) strips tools and injects instructions elsewhere;
  // buildCocoreRequestBody itself only builds the body, so we just verify
  // the body shape is correct (no tools) when family is set.
  const gemmaModel = { ...dummyModel, id: "google/gemma-4-12b" };
  const toolsCtx = {
    messages: [{ role: "user", content: "hi" }],
    systemPrompt: "",
    tools: [{ name: "bash", description: "x", parameters: { type: "object" } }],
  };
  const body = buildCocoreRequestBody(gemmaModel, toolsCtx, "gemma", {}, { routing: "open" });
  assert.equal(body.tools, undefined, "text-tool family must not send OpenAI tools array");
}

await test("verified tools accept only PR #196 model/backend pairings", () => {
  const included = [
    "mlx-community/Qwen3.5-0.8B-MLX-4bit",
    "mlx-community/Qwen3.5-2B-MLX-4bit",
    "mlx-community/Qwen3.5-4B-MLX-4bit",
    "mlx-community/Qwen3.5-9B-MLX-4bit",
    "mlx-community/Qwen3.5-27B-4bit",
    "mlx-community/Qwen3.5-35B-A3B-4bit",
    "mlx-community/Qwen3.5-122B-A10B-4bit",
    "mlx-community/Qwen3.5-397B-A17B-4bit",
    "mlx-community/Qwen3.6-27B-4bit",
    "mlx-community/Qwen3.6-35B-A3B-4bit",
    "mlx-community/Qwen3.6-35B-A3B-4bit-DWQ",
    "mlx-community/Qwen2.5-0.5B-Instruct-4bit",
    "mlx-community/Qwen2.5-3B-Instruct-4bit",
    "mlx-community/Qwen2.5-7B-Instruct-4bit",
    "mlx-community/Qwen2.5-32B-Instruct-4bit",
  ];
  for (const id of included) {
    assert.equal(isVerifiedToolModel(id), true, id);
    assert.equal(getModelFamily(id), null, `${id} uses native tool calls`);
  }
  assert.equal(isVerifiedToolModel("MLX-COMMUNITY/QWEN3.6-27B-4BIT"), true);

  const excluded = [
    "qwen/qwen3.5-9b",
    "qwen/qwen2.5-coder-14b",
    "qwen/Qwen2.5-7B-Instruct-4bit",
    "mlx-community/Qwen2.5-1.5B-Instruct-4bit",
    "mlx-community/Qwen2.5-14B-Instruct-4bit",
    "mlx-community/Qwen2.5-72B-Instruct-4bit",
    "mlx-community/Qwen2.5-0.5B-Instruct",
    "mlx-community/Qwen2.5-3B-Instruct",
    "mlx-community/Qwen3.5-4B-MLX-8bit",
    "mlx-community/Qwen3.6-27B-4bit-GGUF",
    "mlx-community/Llama-3.3-70B-Instruct-4bit",
    "google/gemma-4-12b",
    "deepseek-coder-v2-lite-instruct",
    "mistralai/ministral-3-14b-reasoning",
    "stub",
  ];
  for (const id of excluded) {
    assert.equal(isVerifiedToolModel(id), false, id);
  }
  assert.equal(getModelFamily("qwen/qwen3.5-9b"), "qwen");
  assert.equal(isVerifiedToolModel("leonsarmiento/Ornith-1.0-35B-5bit-mlx"), true);
});

function checkExtensionScenario(scenario) {
  const result = spawnSync(
    process.execPath,
    [
      "--experimental-strip-types",
      "--no-warnings",
      fileURLToPath(new URL("fixtures/cocore-extension.mjs", import.meta.url)),
      scenario,
    ],
    { encoding: "utf8", timeout: 30_000 },
  );
  assert.equal(result.status, 0, result.stdout + result.stderr);
}

await test("catalog models use text/native tools or surface rejection on every tier", () => {
  checkExtensionScenario("tool-postprocessing");
});

await test("legacy configs register four independently routed providers", () => {
  checkExtensionScenario("legacy-routing");
});

await test("registered closures preserve dispatch pins on every tier", () => {
  checkExtensionScenario("routing");
});

await test("setup preserves blanks and clears only explicitly selected pins", () => {
  checkExtensionScenario("setup");
});

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

const streamModel = {
  ...dummyModel,
  provider: "cocore-verified",
  baseUrl: "https://test.invalid/api/v1",
};
const streamContext = {
  ...dummyContext,
  tools: [{ name: "bash", description: "Run a shell command", parameters: { type: "object" } }],
};

await test("HTTP dispatch failures surface actionable errors without forbidden retries", async (t) => {
  const cases = [
    [404, "model_not_found", "invalid_request_error", /reload/],
    [404, "no_friends_for_model", "invalid_request_error", /friends/],
    // A known code must override even a normally retryable HTTP status.
    [503, "insufficient_credits", "insufficient_credits_error", /top up/i],
    [400, "tool_calls_not_supported", "invalid_request_error", /connected.*tool calls/i],
    [401, "onboarding_required", "authentication_error", /onboard/i],
    [401, "authentication_error", "authentication_error", /API key rejected/i],
  ];
  for (const [status, code, type, messagePattern] of cases) {
    await t.test(code, async () => {
      let calls = 0;
      const events = await collectEvents(streamCocore(
        streamModel, streamContext, { apiKey: "fixture-key", maxRetries: 2 },
        {
          routing: "verified", country: "US", minProviderVersion: "0.9.32",
          fetchImpl: async (url, options) => {
            calls++;
            assert.equal(url, "https://test.invalid/api/v1/verified/chat/completions");
            const body = JSON.parse(options.body);
            assert.equal(body.min_trust, "hardware-attested");
            assert.equal(body.country, "US");
            assert.equal(body.min_provider_version, "0.9.32");
            assert.equal(body.tools[0].function.name, "bash");
            return Response.json(
              { error: { type, code, message: "upstream diagnostic" } },
              { status },
            );
          },
        },
      ));
      assert.equal(calls, 1, `${code} must not retry or switch to text-tool requests`);
      assert.equal(events.filter((event) => event.type === "error").length, 1, code);
      assert.equal(events.some((event) => event.type === "done"), false, code);
      assert.equal(events.at(-1).error.stopReason, "error", code);
      assert.match(events.at(-1).error.errorMessage, messagePattern, code);
    });
  }
});

await test("capacity errors retry successfully without changing routing pins", async () => {
  const cases = [
    [503, "no_providers_connected", "service_unavailable_error"],
    [503, "no_friends_available", "service_unavailable_error"],
    [503, "no_providers_for_country", "service_unavailable_error"],
    [503, "no_providers_for_version", "service_unavailable_error"],
    [503, "no_pro_bono_providers", "service_unavailable_error"],
    [502, "pro_bono_lookup_failed", "server_error"],
  ];
  for (const [status, code, type] of cases) {
    const requests = [];
    const events = await collectEvents(streamCocore(
      streamModel, streamContext, { apiKey: "fixture-key", maxRetries: 1 },
      {
        routing: "verified", country: "US", minProviderVersion: "0.9.32",
        fetchImpl: async (url, options) => {
          requests.push({ url, body: JSON.parse(options.body) });
          if (requests.length === 1) {
            return Response.json({ error: { type, code, message: "not available" } }, { status });
          }
          return sseResponse([{ choices: [{ delta: { content: "online now" }, finish_reason: "stop" }] }]);
        },
      },
    ));
    assert.equal(requests.length, 2, `${code} retries once`);
    assert.deepEqual(requests[1], requests[0], `${code} must not weaken routing on retry`);
    assert.equal(events.some((event) => event.type === "error"), false, code);
    assert.equal(events.at(-1).type, "done", code);
    assert.equal(events.at(-1).message.content[0].text, "online now", code);
  }
});

await test("retry exhaustion preserves the friendly verified-provider error", async () => {
  let calls = 0;
  const events = await collectEvents(streamCocore(
    streamModel, streamContext, { apiKey: "fixture-key", maxRetries: 1 },
    {
      routing: "verified",
      fetchImpl: async () => {
        calls++;
        return Response.json({ error: {
          type: "service_unavailable_error", code: "no_verified_providers", message: "opaque diagnostic",
        } }, { status: 503 });
      },
    },
  ));
  assert.equal(calls, 2);
  assert.equal(events.filter((event) => event.type === "error").length, 1);
  assert.equal(events.some((event) => event.type === "done"), false);
  assert.match(events.at(-1).error.errorMessage, /cryptographically verified|attested/i);
  assert.doesNotMatch(events.at(-1).error.errorMessage, /opaque diagnostic/);
});

await test("maxRetries zero surfaces a retryable dispatch failure immediately", async () => {
  let calls = 0;
  const events = await collectEvents(streamCocore(
    streamModel, dummyContext, { maxRetries: 0 },
    { fetchImpl: async () => {
      calls++;
      return Response.json({ error: {
        type: "service_unavailable_error", code: "no_providers_for_country", message: "none in US",
      } }, { status: 503 });
    } },
  ));
  assert.equal(calls, 1);
  assert.equal(events.at(-1).type, "error");
  assert.match(events.at(-1).error.errorMessage, /country/);
});

const gemmaStreamModel = { ...streamModel, id: "google/gemma-4-12b" };
const todoParameters = Type.Object({
  action: Type.Union([Type.Literal("create"), Type.Literal("update")]),
  subject: Type.Optional(Type.String()),
  description: Type.Optional(Type.String()),
  activeForm: Type.Optional(Type.String()),
  id: Type.Optional(Type.Integer()),
  status: Type.Optional(Type.String()),
});
const todoDeclaration = { name: "todo", description: "Manage tasks", parameters: todoParameters };
const reportedCreate = '<|tool_call>call:todo:create{"activeForm":"researching proposal requirements",' +
  'description:"Research and gather information",subject:"Research Defending OSS with AI"}<tool_call|>';
const reportedUpdate = '<|tool_call>call:todo:update{"action":"update","id":1,"status":"in_progress",' +
  '"activeForm":"initial research"}<tool_call|>';

await test("fragmented Gemma text produces tool events before done", async () => {
  const text = "Before. " + reportedCreate + " After.";
  const events = await collectEvents(streamCocore(
    gemmaStreamModel, { messages: [], tools: [todoDeclaration] }, { maxRetries: 0 },
    { fetchImpl: async () => sseResponse([
      { choices: [{ delta: { content: text.slice(0, 20) }, finish_reason: null }] },
      { choices: [{ delta: { content: text.slice(20, 90) }, finish_reason: null }] },
      { choices: [{ delta: { content: text.slice(90) }, finish_reason: "stop" }] },
    ]) },
  ));
  assert.deepEqual(events.map((event) => event.type), [
    "start", "text_start", "text_delta", "text_end",
    "toolcall_start", "toolcall_delta", "toolcall_end",
    "text_start", "text_delta", "text_end", "done",
  ]);
  const done = events.at(-1);
  assert.equal(done.reason, "toolUse");
  assert.equal(done.message.content[0].text, "Before. ");
  assert.equal(done.message.content[2].text, " After.");
  const call = done.message.content[1];
  assert.equal(call.name, "todo");
  assert.deepEqual(call.arguments, {
    action: "create", activeForm: "researching proposal requirements",
    description: "Research and gather information", subject: "Research Defending OSS with AI",
  });
  const end = events.find((event) => event.type === "toolcall_end");
  assert.equal(end.contentIndex, 1);
  assert.deepEqual(end.toolCall, call);
  assert.deepEqual(JSON.parse(events.find((event) => event.type === "toolcall_delta").delta), call.arguments);
});

await test("Gemma refuses unavailable tools and truncated or invalid calls", async (t) => {
  for (const [label, text, tools, finishReason] of [
    ["no tools", reportedCreate, [], "stop"],
    ["unknown tool", '<|tool_call>call:unknown{"action":"create"}<tool_call|>', [todoDeclaration], "stop"],
    ["conflicting action", '<|tool_call>call:todo:update{"action":"create"}<tool_call|>', [todoDeclaration], "stop"],
    ["invalid arguments", '<|tool_call>call:todo:update{"id":"not a number"}<tool_call|>', [todoDeclaration], "stop"],
    ["broken JSON", '<|tool_call>call:todo:create{subject:unfinished}<tool_call|>', [todoDeclaration], "stop"],
    ["token limit", '<|tool_call>call:todo:create{"activeForm":"researching proposal requirements",' +
      'description:"Research and gather information",subject:"Research Defending OSS with AI"', [todoDeclaration], "length"],
    ["reasoning only", '<think>' + reportedCreate + '</think>Finished.', [todoDeclaration], "stop"],
  ]) {
    await t.test(label, async () => {
      const events = await collectEvents(streamCocore(
        gemmaStreamModel, { messages: [], tools }, { maxRetries: 0 },
        { fetchImpl: async () => sseResponse([{ choices: [{ delta: { content: text }, finish_reason: finishReason }] }]) },
      ));
      assert.equal(events.some((event) => event.type === "toolcall_end"), false);
      assert.equal(events.at(-1).message.content.some((block) => block.type === "toolCall"), false);
      assert.equal(events.at(-1).reason, finishReason);
    });
  }
});

await test("aborted Gemma text never becomes executable calls", async () => {
  const controller = new AbortController();
  const events = await collectEvents(streamCocore(
    gemmaStreamModel, { messages: [], tools: [todoDeclaration] },
    { maxRetries: 0, signal: controller.signal },
    { fetchImpl: async () => {
      const response = sseResponse([{ choices: [{ delta: { content: reportedCreate }, finish_reason: "stop" }] }]);
      controller.abort();
      return response;
    } },
  ));
  assert.equal(events.at(-1).type, "error");
  assert.equal(events.at(-1).reason, "aborted");
  assert.equal(events.some((event) => event.type === "toolcall_end"), false);
});

await test("Gemma does not duplicate native tool calls with identical text envelopes", async () => {
  // Native todo:update carries the same effective args as the text envelope.
  // The dedup path (post fix) should drop the text-extracted call so we
  // end with the single native one. This used to be guaranteed by the
  // skip-the-fixer-when-native-calls-exist guard; the new code does it
  // via (name, args) comparison instead, which is the load-bearing
  // invariant.
  const nativeArgs = '{"action":"update","id":1,"status":"in_progress","activeForm":"initial research"}';
  const events = await collectEvents(streamCocore(
    gemmaStreamModel, { messages: [], tools: [todoDeclaration] }, { maxRetries: 0 },
    { fetchImpl: async () => sseResponse([{ choices: [{ delta: {
      content: reportedUpdate,
      tool_calls: [{ index: 0, id: "native-todo", type: "function", function: {
        name: "todo", arguments: nativeArgs,
      } }],
    }, finish_reason: "tool_calls" }] }]) },
  ));
  const calls = events.at(-1).message.content.filter((block) => block.type === "toolCall");
  assert.equal(calls.length, 1);
  assert.equal(calls[0].id, "native-todo");
  assert.equal(events.filter((event) => event.type === "toolcall_end").length, 1);
  assert.equal(events.at(-1).reason, "toolUse");
});

const fetchDeclaration = {
  name: "fetch_content",
  description: "Fetch a URL",
  parameters: {
    type: "object",
    properties: { url: { type: "string" } },
    required: ["url"],
  },
};
const writeDeclaration = {
  name: "write",
  description: "Write a file",
  parameters: {
    type: "object",
    properties: {
      path: { type: "string" },
      content: { type: "string" },
    },
    required: ["path", "content"],
  },
};
const writeCall = '<|tool_call>call:write{"path":"/tmp/oss.md","content":"# report"}<|tool_call|>';

await test("Gemma keeps native and text tool calls when their names differ", async () => {
  // Reproduces the 1b9d828c failure: model emits native fetch_content for
  // qodo.ai and a text write call to persist the report. The two calls
  // are different; both must execute. Before the fix, the text write
  // was silently dropped because the guard skipped fixCocoreToolCalls
  // when toolCallAccumulators.size > 0.
  const events = await collectEvents(streamCocore(
    gemmaStreamModel, { messages: [], tools: [fetchDeclaration, writeDeclaration] },
    { maxRetries: 0 },
    { fetchImpl: async () => sseResponse([{ choices: [{ delta: {
      content: writeCall,
      tool_calls: [{ index: 0, id: "native-fetch", type: "function", function: {
        name: "fetch_content", arguments: '{"url":"https://qodo.ai/pricing/"}',
      } }],
    }, finish_reason: "tool_calls" }] }]) },
  ));
  const calls = events.at(-1).message.content.filter((block) => block.type === "toolCall");
  assert.equal(calls.length, 2);
  const byName = Object.fromEntries(calls.map((call) => [call.name, call]));
  assert.equal(byName.fetch_content.id, "native-fetch");
  assert.equal(byName.fetch_content.arguments.url, "https://qodo.ai/pricing/");
  assert.equal(byName.write.name, "write");
  assert.equal(byName.write.arguments.path, "/tmp/oss.md");
  assert.equal(byName.write.arguments.content, "# report");
  assert.equal(events.filter((event) => event.type === "toolcall_end").length, 2);
  assert.equal(events.at(-1).reason, "toolUse");
});

await test("Gemma keeps native and text tool calls when arguments differ", async () => {
  // Two text-extracted todo:update envelopes with different ids must
  // both survive even though a native todo:update is present. Same
  // (name, args) dedup, not name-only dedup.
  const differentUpdate =
    '<|tool_call>call:todo:update{"action":"update","id":2,"status":"in_progress",' +
    '"activeForm":"second pass"}<|tool_call|>';
  const events = await collectEvents(streamCocore(
    gemmaStreamModel, { messages: [], tools: [todoDeclaration] }, { maxRetries: 0 },
    { fetchImpl: async () => sseResponse([{ choices: [{ delta: {
      content: differentUpdate,
      tool_calls: [{ index: 0, id: "native-todo", type: "function", function: {
        name: "todo", arguments: '{"action":"update","id":1,"status":"in_progress"}',
      } }],
    }, finish_reason: "tool_calls" }] }]) },
  ));
  const calls = events.at(-1).message.content.filter((block) => block.type === "toolCall");
  assert.equal(calls.length, 2);
  const byId = Object.fromEntries(calls.map((call) => [call.id, call]));
  assert.equal(byId["native-todo"].arguments.id, 1);
  const textCall = calls.find((call) => call.id !== "native-todo");
  assert.ok(textCall, "text-extracted call must exist with a fresh id");
  assert.equal(textCall.name, "todo");
  assert.equal(textCall.arguments.id, 2);
  assert.equal(events.filter((event) => event.type === "toolcall_end").length, 2);
  assert.equal(events.at(-1).reason, "toolUse");
});

await test("Gemma still extracts a complete text tool call when the response is truncated", async () => {
  // The old guard skipped fixCocoreToolCalls on stopReason === "length"
  // out of caution. The new code runs the fixer anyway: a complete
  // <|tool_call|>…<|tool_call|> envelope with parseable JSON is safe to
  // execute. Truncation only excludes unclosed envelopes (next test).
  const events = await collectEvents(streamCocore(
    gemmaStreamModel, { messages: [], tools: [writeDeclaration] }, { maxRetries: 0 },
    { fetchImpl: async () => sseResponse([{ choices: [{ delta: {
      content: writeCall,
    }, finish_reason: "length" }] }]) },
  ));
  const calls = events.at(-1).message.content.filter((block) => block.type === "toolCall");
  assert.equal(calls.length, 1);
  assert.equal(calls[0].name, "write");
  assert.equal(calls[0].arguments.path, "/tmp/oss.md");
  assert.equal(events.at(-1).reason, "toolUse");
});

await test("Gemma ignores unclosed text tool calls even when the response is truncated", async () => {
  // Mirrors the 1b9d828c final turn: the model emits an opener but the
  // chat template stops the model before the close delimiter. The
  // parser cannot match an unclosed envelope, so no tool call is
  // produced and no exception is thrown. The remaining text is surfaced
  // as the assistant's final message so the supervisor can see what the
  // model actually said.
  const truncated = '<|tool_call>call:write{"path":"/tmp/oss.md","content":"# report';
  const events = await collectEvents(streamCocore(
    gemmaStreamModel, { messages: [], tools: [writeDeclaration] }, { maxRetries: 0 },
    { fetchImpl: async () => sseResponse([{ choices: [{ delta: {
      content: truncated,
    }, finish_reason: "stop" }] }]) },
  ));
  const calls = events.at(-1).message.content.filter((block) => block.type === "toolCall");
  assert.equal(calls.length, 0);
  assert.equal(events.some((event) => event.type === "toolcall_end"), false);
  const text = events.at(-1).message.content.find((block) => block.type === "text");
  assert.ok(text, "unmatched text must remain visible to the caller");
  assert.equal(text.text, truncated);
  assert.equal(events.at(-1).reason, "stop");
});

await test("Gemma retries discard unfinished text from the failed attempt", async () => {
  let requests = 0;
  const events = await collectEvents(streamCocore(
    gemmaStreamModel, { messages: [], tools: [todoDeclaration] }, { maxRetries: 1 },
    { fetchImpl: async () => {
      requests++;
      if (requests === 1) {
        let reads = 0;
        return new Response(new ReadableStream({
          pull(controller) {
            if (reads++ === 0) {
              const chunk = { choices: [{ delta: { content: reportedCreate }, finish_reason: null }] };
              controller.enqueue(new TextEncoder().encode(`data: ${JSON.stringify(chunk)}\n\n`));
            } else controller.error(new Error("fixture stream disconnected"));
          },
        }));
      }
      return sseResponse([{ choices: [{ delta: { content: reportedUpdate }, finish_reason: "stop" }] }]);
    } },
  ));
  assert.equal(requests, 2);
  const calls = events.at(-1).message.content.filter((block) => block.type === "toolCall");
  assert.equal(calls.length, 1, "failed-attempt create must not execute");
  assert.deepEqual(calls[0].arguments, {
    action: "update", id: 1, status: "in_progress", activeForm: "initial research",
  });
  assert.equal(events.filter((event) => event.type === "toolcall_end").length, 1);
});

await test("real Pi agent executes recovered Gemma todo calls and continues", { timeout: 10_000 }, async () => {
  // Resolve Pi's own agent-core dependency without assuming npm hoists it.
  const hostRequire = createRequire(import.meta.resolve("@earendil-works/pi-coding-agent"));
  const corePackage = hostRequire.resolve("@earendil-works/pi-agent-core/package.json");
  const { Agent } = await import(new URL("./dist/index.js", pathToFileURL(corePackage)));
  const tasks = [];
  let requests = 0;
  const agent = new Agent({
    initialState: {
      model: gemmaStreamModel,
      tools: [{
        ...todoDeclaration, label: "Tasks",
        execute: async (_id, args) => {
          if (args.action === "create") tasks.push({ id: 1, subject: args.subject, status: "pending" });
          else tasks.find((task) => task.id === args.id).status = args.status;
          return { content: [{ type: "text", text: "Task saved" }], details: {} };
        },
      }],
    },
    streamFn: (model, context, options) => streamCocore(model, {
      // Pi 0.87 carries declarations in system messages; the extension API uses legacy Context.
      systemPrompt: getCurrentSystemPrompt(context.messages),
      tools: getCurrentTools(context.messages),
      messages: context.messages.filter((message) => message.role !== "system"),
    }, { ...options, maxRetries: 0 }, {
      fetchImpl: async (_url, options) => {
        requests++;
        assert.ok(requests <= 3, "agent must stop after its final text response");
        const body = JSON.parse(options.body);
        assert.equal(body.tools, undefined);
        if (requests > 1) assert.match(JSON.stringify(body.messages), /Task saved/);
        return sseResponse([{ choices: [{ delta: {
          content: [reportedCreate, reportedUpdate, "Research can begin."][requests - 1],
        }, finish_reason: "stop" }] }]);
      },
    }),
  });
  await agent.prompt("Create a research task and start it.");
  assert.deepEqual(tasks, [{ id: 1, subject: "Research Defending OSS with AI", status: "in_progress" }]);
  assert.equal(requests, 3);
  const results = agent.state.messages.filter((message) => message.role === "toolResult");
  assert.equal(results.length, 2);
  assert.equal(results.every((message) => message.toolName === "todo" && !message.isError), true);
  const calls = agent.state.messages.filter((message) => message.role === "assistant")
    .flatMap((message) => message.content.filter((block) => block.type === "toolCall"));
  assert.deepEqual(results.map((message) => message.toolCallId), calls.map((call) => call.id));
});

await test("native tool-call fragments keep IDs, arguments, and toolUse finish reason", async () => {
  const events = await collectEvents(streamCocore(
    streamModel, streamContext, { maxRetries: 0 },
    { fetchImpl: async () => sseResponse([
      { choices: [{ delta: { tool_calls: [{
        index: 0, id: "call-1", type: "function",
        function: { name: "bash", arguments: '{"command":' },
      }] }, finish_reason: null }] },
      { choices: [{ delta: { tool_calls: [{ index: 0, function: { arguments: '"pwd"}' } }] }, finish_reason: null }] },
      { choices: [{ delta: {}, finish_reason: "tool_calls" }] },
    ]) },
  ));
  assert.equal(events.at(-1).reason, "toolUse");
  assert.deepEqual(events.at(-1).message.content, [{
    type: "toolCall", id: "call-1", name: "bash", arguments: { command: "pwd" },
  }]);
  assert.equal(events.filter((event) => event.type === "toolcall_end").length, 1);
});

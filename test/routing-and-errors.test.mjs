// Self-check for the upstream-API surface introduced for cocore.dev
// routing tiers, dispatch-error mapping, and the verified-tool
// allowlist (PR #196). Run with:
//
//   node --experimental-strip-types --no-warnings \
//     test/routing-and-errors.test.mjs

import assert from "node:assert/strict";
import { test } from "node:test";
import { spawnSync } from "node:child_process";
import { fileURLToPath } from "node:url";
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

await test("text-tool responses are postprocessed on every routing tier", () => {
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

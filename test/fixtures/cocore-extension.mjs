import assert from "node:assert/strict";
import { mkdtempSync, mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

// A separate process isolates the extension's config path and handler guard.
const home = mkdtempSync(join(tmpdir(), "pi-cocore-test-"));
process.env.HOME = home;
const configDir = join(home, ".pi", "agent");
mkdirSync(configDir, { recursive: true });
const scenario = process.argv[2];
const configPath = join(configDir, "cocore-config.json");
const initialConfig = { apiKey: "fixture-key" };
if (scenario === "routing" || scenario === "setup") {
  initialConfig.country = "US";
  initialConfig.minProviderVersion = "v0.9.32";
}
writeFileSync(configPath, JSON.stringify(initialConfig));

const { default: extension } = await import("../../extensions/cocore.ts");
const providers = [];
const handlers = new Map();
const commands = new Map();
const pi = {
  registerProvider: (key, config) => providers.push({ key, ...config }),
  on: (name, handler) => handlers.set(name, handler),
  registerCommand: (name, command) => commands.set(name, command),
};
const tools = [{
  name: "bash",
  description: "Run a shell command",
  parameters: { type: "object", properties: { command: { type: "string" } } },
}];
const context = { messages: [{ role: "user", content: "run pwd" }], tools };
const gemmaId = "google/gemma-4-12b";
const nativeId = "mlx-community/Qwen3.5-4B-MLX-4bit";
const textQwenId = "qwen/qwen3.5-9b";
const nativeIds = [
  nativeId,
  "mlx-community/Qwen2.5-7B-Instruct-4bit",
  "mlx-community/Qwen3.5-0.8B-MLX-4bit",
  "mlx-community/Qwen3.5-9B-MLX-4bit",
];
const bonsaiId = "prism-ml/Ternary-Bonsai-27B-mlx-2bit";
const catalogIds = [
  ...nativeIds,
  gemmaId,
  "mlx-community/gemma-3-4b-it-qat-4bit",
  "mlx-community/gemma-4-12B-it-8bit",
  bonsaiId,
  textQwenId,
];
let responseId;
const requests = [];

function sse(delta, finishReason = "stop") {
  return new Response(
    `data: ${JSON.stringify({ choices: [{ delta, finish_reason: finishReason }] })}\n\n` +
      "data: [DONE]\n\n",
    { headers: { "Content-Type": "text/event-stream" } },
  );
}

globalThis.fetch = async (url, options) => {
  if (url === "https://cocore.dev/api/v1/models") {
    assert.equal(options.headers.Authorization, "Bearer fixture-key");
    return Response.json({
      object: "list",
      data: [...catalogIds, "stub"].map((id) => ({
        id, object: "model", created: 1, owned_by: "cocore",
      })),
    });
  }
  const body = JSON.parse(options.body);
  requests.push({ url, body, headers: options.headers });
  responseId = body.model;
  if (body.model === bonsaiId) {
    assert.equal(body.tools[0].function.name, "bash", "Bonsai needs upstream native support");
    return Response.json({ error: {
      code: "tool_calls_not_supported",
      message: "No connected provider supports tool calling for this model",
    } }, { status: 400 });
  }
  if (nativeIds.includes(body.model)) {
    return sse({ tool_calls: [{
      index: 0,
      id: "call-native",
      type: "function",
      function: { name: "bash", arguments: '{"command":"pwd"}' },
    }] }, "tool_calls");
  }
  assert.equal(body.tools, undefined, "text-tool requests must not send native tools");
  assert.match(body.messages[0].content, /bash/);
  return sse({ content: body.model === textQwenId
    ? '<tool_call>\n{"name":"bash","arguments":{"command":"pwd"}}\n</tool_call>'
    : '<|tool_call|>bash{"command":"pwd"}',
  });
};

await extension(pi);
assert.deepEqual(providers.map((p) => p.key), [
  "cocore", "cocore-private", "cocore-verified", "cocore-probono",
]);
assert.equal(new Set(providers.map((p) => p.streamSimple)).size, 4);
assert.equal(new Set(providers.map((p) => p.models)).size, 1);
assert.deepEqual(providers[0].models.map((m) => m.id), catalogIds);

if (scenario === "tool-postprocessing") {
  for (const modelId of catalogIds) {
    for (const provider of providers) {
      const model = {
        ...provider.models.find((m) => m.id === modelId),
        api: provider.api,
        provider: provider.key,
        baseUrl: provider.baseUrl,
      };
      const events = [];
      for await (const event of provider.streamSimple(
        model, context, { apiKey: "fixture-key", maxRetries: 0 },
      )) events.push(event);
      assert.equal(responseId, modelId);
      if (modelId === bonsaiId) {
        assert.equal(events.at(-1).type, "error", "unsupported models must not claim tool success");
        assert.match(events.at(-1).error.errorMessage, /connected.*tool calls/i);
        assert.equal(events.some((event) => event.type === "toolcall_end"), false);
        continue;
      }
      const done = events.find((event) => event.type === "done");
      assert.ok(done, `${provider.key}/${modelId} completes`);
      assert.equal(done.reason, "toolUse", "the stream must request tool execution");
      assert.equal(events.filter((event) => event.type === "toolcall_end").length, 1);
      if (nativeIds.includes(modelId)) {
        assert.equal(done.reason, "toolUse");
        assert.deepEqual(done.message.content[0], {
          type: "toolCall", id: "call-native", name: "bash", arguments: { command: "pwd" },
        });
      } else {
        const call = done.message.content.find((block) => block.type === "toolCall");
        assert.equal(call?.name, "bash", `${provider.key}/${modelId} extracts bash`);
        assert.deepEqual(call.arguments, { command: "pwd" });
      }
    }
  }
  assert.equal(handlers.has("message_end"), false, "conversion belongs to the provider stream");
} else if (scenario === "routing" || scenario === "legacy-routing") {
  const expectedPaths = [
    "/chat/completions", "/private/chat/completions",
    "/verified/chat/completions", "/probono/chat/completions",
  ];
  for (const [index, provider] of providers.entries()) {
    const model = {
      ...provider.models.find((m) => m.id === nativeId),
      api: provider.api, provider: provider.key, baseUrl: provider.baseUrl,
    };
    const events = [];
    for await (const event of provider.streamSimple(
      model, context, { apiKey: "fixture-key", maxRetries: 0 },
    )) events.push(event);
    const request = requests.at(-1);
    assert.equal(request.url, `https://cocore.dev/api/v1${expectedPaths[index]}`);
    assert.equal(request.headers.Authorization, "Bearer fixture-key");
    assert.equal(request.body.country, initialConfig.country);
    assert.equal(request.body.min_provider_version, scenario === "routing" ? "0.9.32" : undefined);
    assert.equal(request.body.min_trust, index === 2 ? "hardware-attested" : undefined);
    assert.equal(Object.hasOwn(request.body, "minProviderVersion"), false);
    assert.equal(Object.hasOwn(request.body, "minTrust"), false);
    assert.deepEqual(request.body.tools, [{ type: "function", function: tools[0] }]);
    assert.equal(events.at(-1).reason, "toolUse");
    assert.equal(events.at(-1).message.provider, provider.key);
  }
  assert.equal(requests.length, 4);
} else if (scenario === "setup") {
  const command = commands.get("cocore-setup");
  const notifications = [];
  async function setup(inputs, expected) {
    const answers = [...inputs];
    await command.handler("", { ui: {
      input: async () => answers.shift(),
      notify: (message) => notifications.push(message),
    } });
    assert.equal(answers.length, 0, "setup must consume the three inputs");
    assert.deepEqual(JSON.parse(readFileSync(configPath, "utf8")), expected);
    assert.match(notifications.at(-1), /reload/);
  }
  await setup(["  ", "  ", "  "], initialConfig);
  await setup(["replacement-key", "", ""], {
    ...initialConfig, apiKey: "replacement-key",
  });
  await setup(["", " CA ", " 0.9.51 "], {
    apiKey: "replacement-key", country: "CA", minProviderVersion: "0.9.51",
  });
  await setup(["", " - ", ""], {
    apiKey: "replacement-key", minProviderVersion: "0.9.51",
  });
  await setup(["", "", "-"], { apiKey: "replacement-key" });
  await setup(["", "US", "v0.9.32"], {
    apiKey: "replacement-key", country: "US", minProviderVersion: "v0.9.32",
  });
  await setup(["final-key", "-", "-"], { apiKey: "final-key" });
  await setup(["", "-", "-"], { apiKey: "final-key" });
} else {
  throw new Error(`Unknown extension scenario: ${scenario}`);
}

console.log(`ok - extension ${scenario} passed`);

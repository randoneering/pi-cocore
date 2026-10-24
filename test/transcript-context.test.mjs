import assert from "node:assert/strict";
import { normalizeContext } from "@earendil-works/pi-ai";
import { test } from "node:test";
import { buildCocoreRequestBody, streamCocore } from "../extensions/cocore.ts";

const nativeModel = {
  id: "mlx-community/Qwen3.5-4B-MLX-4bit",
  api: "openai-completions",
  provider: "cocore",
  baseUrl: "https://test.invalid/api/v1",
};
const gemmaModel = { ...nativeModel, id: "google/gemma-4-12b" };
const bash = {
  name: "bash",
  description: "Run a shell command",
  parameters: { type: "object", properties: { command: { type: "string" } } },
};
const route = { routing: "open" };

function sse(chunks) {
  return new Response(chunks.map((chunk) => `data: ${JSON.stringify(chunk)}\n\n`).join("") + "data: [DONE]\n\n", {
    headers: { "Content-Type": "text/event-stream" },
  });
}

async function events(stream) {
  const result = [];
  for await (const event of stream) result.push(event);
  return result;
}

await test("normalized context preserves system prompt and native tools in request", () => {
  const context = normalizeContext({
    systemPrompt: "SYSTEM_MARKER",
    tools: [bash],
    messages: [{ role: "user", content: "run pwd" }],
  });
  const body = buildCocoreRequestBody(nativeModel, context, null, {}, route);
  assert.equal(body.messages[0].role, "system");
  assert.equal(body.messages[0].content, "SYSTEM_MARKER");
  assert.deepEqual(body.tools, [{ type: "function", function: bash }]);
});

await test("registered streamSimple callback preserves normalized system prompt and native tools", async () => {
  const context = normalizeContext({
    systemPrompt: "SYSTEM_MARKER",
    tools: [bash],
    messages: [{ role: "user", content: "run pwd" }],
  });
  let request;
  const result = await events(streamCocore(nativeModel, context, { apiKey: "fixture", maxRetries: 0 }, {
    fetchImpl: async (_url, options) => {
      request = JSON.parse(options.body);
      return sse([{ choices: [{ delta: { content: "done" }, finish_reason: "stop" }] }]);
    },
  }));
  assert.equal(result.at(-1).type, "done");
  assert.equal(request.messages[0].content, "SYSTEM_MARKER");
  assert.deepEqual(request.tools, [{ type: "function", function: bash }]);
});

await test("tools-only legacy Context is normalized for Gemma text-tool extraction", async () => {
  const context = { messages: [{ role: "user", content: "run pwd" }], tools: [bash] };
  let request;
  const result = await events(streamCocore(gemmaModel, context, { apiKey: "fixture", maxRetries: 0 }, {
    fetchImpl: async (_url, options) => {
      request = JSON.parse(options.body);
      return sse([{ choices: [{ delta: { content: '<|tool_call|>bash{"command":"pwd"}' }, finish_reason: "stop" }] }]);
    },
  }));
  assert.equal(request.tools, undefined);
  assert.match(request.messages[0].content, /bash/);
  assert.equal(result.at(-1).reason, "toolUse");
  assert.deepEqual(result.at(-1).message.content.find((block) => block.type === "toolCall").arguments,
    { command: "pwd" });
});

await test("normalized transcript later system updates and tool removal are reflected", () => {
  const added = { name: "read", description: "Read a file", parameters: { type: "object" } };
  const removed = { name: "bash" };
  const context = normalizeContext({
    systemPrompt: "SYSTEM_MARKER",
    tools: [bash],
    messages: [
      { role: "user", content: "first" },
      { role: "system", content: "LATER_MARKER", toolsAdded: [added], toolsRemoved: [removed], timestamp: 1 },
      { role: "user", content: "second" },
    ],
  });
  const body = buildCocoreRequestBody(nativeModel, context, null, {}, route);
  assert.equal(body.messages[0].content, "SYSTEM_MARKER\n\nLATER_MARKER");
  assert.deepEqual(body.tools.map((tool) => tool.function.name), ["read"]);
});

await test("normalized context injects current tools into Gemma prompt without native tool field", async () => {
  const context = normalizeContext({
    systemPrompt: "SYSTEM_MARKER",
    tools: [bash],
    messages: [{ role: "user", content: "run pwd" }],
  });
  let request;
  const result = await events(streamCocore(gemmaModel, context, { apiKey: "fixture", maxRetries: 0 }, {
    fetchImpl: async (_url, options) => {
      request = JSON.parse(options.body);
      return sse([{ choices: [{ delta: { content: "done" }, finish_reason: "stop" }] }]);
    },
  }));
  assert.equal(result.at(-1).type, "done");
  assert.equal(request.tools, undefined);
  assert.match(request.messages[0].content, /SYSTEM_MARKER/);
  assert.match(request.messages[0].content, /bash/);
});

await test("normalized transcript system prompt and tools survive completion continuation", async () => {
  const context = normalizeContext({
    systemPrompt: "SYSTEM_MARKER",
    tools: [bash],
    messages: [{ role: "user", content: "write a complete response" }],
  });
  const requests = [];
  const result = await events(streamCocore(nativeModel, context, { apiKey: "fixture", maxRetries: 0 }, {
    fetchImpl: async (_url, options) => {
      requests.push(JSON.parse(options.body));
      return requests.length === 1
        ? sse([{ choices: [{ delta: { content: "This response ends abruptly with" }, finish_reason: "stop" }] }])
        : sse([{ choices: [{ delta: { content: " a complete ending." }, finish_reason: "stop" }] }]);
    },
  }));
  assert.equal(result.at(-1).type, "done");
  assert.equal(requests.length, 2, "truncated response should use exactly one continuation request");
  assert.equal(requests[1].messages.find((message) => message.role === "system").content, "SYSTEM_MARKER");
  assert.deepEqual(requests[1].tools, [{ type: "function", function: bash }]);
  assert.ok(requests[1].messages.some((message) =>
    message.role === "assistant" && JSON.stringify(message.content).includes("This response ends abruptly with")));
  assert.equal(requests[1].messages.at(-1).content,
    "Continue your previous response from where it was cut off. Do not repeat any earlier content; just emit the remaining text, closing any open strings, brackets, or words.");
});

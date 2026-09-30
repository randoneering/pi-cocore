import type { ExtensionAPI } from "@earendil-works/pi-coding-agent";
import {
  createAssistantMessageEventStream,
  calculateCost,
  type AssistantMessageEventStream,
  type Context,
  type Model,
  type SimpleStreamOptions,
  type Api,
  type AssistantMessage,
  type StopReason,
  type ToolCall,
  type Message,
  type Tool,
  type Usage,
} from "@earendil-works/pi-ai";
import { existsSync, readFileSync, writeFileSync, mkdirSync } from "node:fs";
import { join } from "node:path";
import { homedir } from "node:os";

// ── Constants ────────────────────────────────────────────────────────────────

/**
 * Canonical cocore.dev base URL. The older `console.cocore.dev` host
 * still serves the same endpoints but docs canonicalize on this one —
 * keeping the strings in sync avoids drift if one host is retired.
 */
const BASE_URL_ROOT = "https://cocore.dev/api/v1";
const BASE_URL = BASE_URL_ROOT;
const CONFIG_DIR = join(homedir(), ".pi", "agent");
const CONFIG_PATH = join(CONFIG_DIR, "cocore-config.json");

// ── Routing tiers ────────────────────────────────────────────────────────────

/**
 * cocore.dev exposes four chat-completions routes that share the same
 * request body but differ in provider selection:
 *
 *   open     — any online provider
 *   private  — providers on your friends list (DID-based trust)
 *   verified — only cryptographically-attested providers
 *   probono  — providers that opt to serve you for free
 *
 * Each tier has its own pi provider entry so users see them as distinct
 * routing choices in the model picker.
 */
type CocoreRouting = "open" | "private" | "verified" | "probono";

const ROUTINGS: CocoreRouting[] = ["open", "private", "verified", "probono"];

/** URL path for the chat-completions endpoint on each routing tier. */
function chatCompletionsPath(routing: CocoreRouting): string {
  return routing === "open" ? "/chat/completions" : `/${routing}/chat/completions`;
}

/** Provider key and display name for a routing tier. */
function providerKey(routing: CocoreRouting): string {
  return routing === "open" ? "cocore" : `cocore-${routing}`;
}

function providerName(routing: CocoreRouting): string {
  return routing === "open" ? "Co/Core" : `Co/Core (${routing})`;
}

// ── Retry configuration ──────────────────────────────────────────────────────

/** Maximum number of retry attempts for failed requests. */
const MAX_RETRIES = 3;

/** Base delay in ms for exponential backoff. */
const RETRY_BASE_DELAY_MS = 1000;

/** HTTP status codes that should be retried. */
const RETRYABLE_STATUS_CODES = new Set([
  408, // Request Timeout
  425, // Too Early
  429, // Too Many Requests
  500, // Internal Server Error
  502, // Bad Gateway
  503, // Service Unavailable
  504, // Gateway Timeout
]);

/**
 * Dispatch-level error codes that the upstream API returns in
 * `body.error.code`. Capacity-shaped ones (no providers online, no
 * providers matching the country pin, etc.) are retryable per the
 * dispatch-errors doc; the others fail closed and surface immediately.
 */
const RETRYABLE_DISPATCH_CODES = new Set([
  "no_providers_connected",
  "no_providers_for_country",
  "no_providers_for_version",
  "no_friends_available",
  "no_pro_bono_providers",
  "pro_bono_lookup_failed",
]);

const NON_RETRYABLE_DISPATCH_CODES = new Set([
  "model_not_found",
  "no_friends_for_model",
  "insufficient_credits",
  "tool_calls_not_supported",
  "onboarding_required",
  "authentication_error",
  "invalid_request_error",
]);

/**
 * Check whether an HTTP-status-level error should trigger a retry.
 * Substring heuristics catch idle-timeout and generic timeouts that
 * ride on a 200-with-empty-body or a non-listed status code; the
 * upstream-API dispatch codes are handled separately by
 * `parseServerError` and considered before this fallback runs.
 */
function isRetryableError(status: number, errorMessage?: string): boolean {
  if (RETRYABLE_STATUS_CODES.has(status)) return true;
  if (errorMessage) {
    const lower = errorMessage.toLowerCase();
    if (lower.includes("idle-timeout") || lower.includes("idle_timeout")) return true;
    if (lower.includes("timeout") || lower.includes("timed out")) return true;
    if (lower.includes("rate limit") || lower.includes("too many requests")) return true;
  }
  return false;
}

/**
 * Pull the dispatch error code (e.g. `model_not_found`,
 * `insufficient_credits`) out of a server error body. Returns null
 * when the body isn't shaped like a cocore API error envelope.
 */
function extractErrorCode(body: string): string | null {
  try {
    const parsed = JSON.parse(body) as { error?: { code?: unknown } } | null;
    const code = parsed?.error?.code;
    if (typeof code === "string" && code.length > 0) return code;
  } catch {
    // Not JSON — leave the code null and let HTTP-status heuristics apply.
  }
  return null;
}

/**
 * Map a server error (HTTP status + body) to a structured envelope the
 * stream loop can act on: a retry decision and a user-facing message.
 * Dispatch codes take precedence over status-code heuristics because
 * the API documents capacity vs. fail-closed behavior per code.
 */
function parseServerError(
  status: number,
  errorText: string,
  modelId?: string,
): { code: string | null; retryable: boolean; friendly: string; raw: string } {
  const code = extractErrorCode(errorText);
  const raw = errorText.slice(0, 300);

  let retryable: boolean;
  if (code !== null) {
    if (RETRYABLE_DISPATCH_CODES.has(code)) {
      retryable = true;
    } else if (NON_RETRYABLE_DISPATCH_CODES.has(code)) {
      retryable = false;
    } else {
      // Unknown dispatch code — fall back to HTTP status + heuristic.
      retryable = isRetryableError(status, errorText);
    }
  } else {
    retryable = isRetryableError(status, errorText);
  }

  const friendly = friendlyMessageFor(code, status, raw, modelId);
  return { code, retryable, friendly, raw };
}

/**
 * Translate a server error into something a human can act on. Each
 * branch points the user at the next concrete step (top up credits,
 * refresh the model list, switch routing tier, etc.) instead of dumping
 * the raw upstream message.
 */
function friendlyMessageFor(
  code: string | null,
  status: number,
  raw: string,
  modelId?: string,
): string {
  const where = (code ?? `${status}`) as string;
  switch (code) {
    case "insufficient_credits":
      return "Not enough co/core credits for this request. Top up at cocore.dev/account, then retry.";
    case "model_not_found":
      return modelId
        ? `Model ${modelId} isn't currently served on co/core. Run /reload to refresh the model list, or pick another model.`
        : "Requested model isn't currently served on co/core. Run /reload to refresh the model list, or pick another model.";
    case "tool_calls_not_supported":
      return "No connected provider currently supports tool calls for this model. Pick another model with live tool support, disable tools, or retry when a capable provider is online.";
    case "onboarding_required":
      return "Your account isn't connected to co/core yet — visit cocore.dev to complete onboarding, then retry.";
    case "authentication_error":
      return "API key rejected — generate a new one at cocore.dev/account and run /cocore-setup.";
    case "no_friends_for_model":
      return "No friends in your network serve this model. Pick a model from a friend or switch routing tier.";
    case "no_friends_available":
      return "No friends in your network are online. Try again later, or switch to a non-private routing tier.";
    case "no_providers_for_country":
      return "No providers in the configured region serve this model. Run /cocore-setup and enter - for country to clear the pin, or pick a different model.";
    case "no_providers_for_version":
      return "No providers run the required tray release for this model. Run /cocore-setup and enter - for minimum provider version to clear the pin, or retry as the fleet updates.";
    case "no_providers_connected":
      return "No co/core providers are online right now. Retrying in a moment usually works.";
    case "no_verified_providers":
      return "No cryptographically verified providers qualify for this request. Retry when an attested provider is online, or explicitly choose another routing tier.";
    case "no_pro_bono_providers":
      return "No connected provider currently serves you for free. Switch to the open or verified routing tier.";
    case "pro_bono_lookup_failed":
      return "The pro-bono provider lookup failed upstream. Retrying usually works.";
    default:
      return `co/core ${where}: ${raw || "no body"}`.trim();
  }
}

/**
 * Sleep for `ms` milliseconds.
 */
function sleep(ms: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

/**
 * Calculate the backoff delay for a given retry attempt.
 * Uses exponential backoff with jitter: base * 2^attempt + random jitter.
 */
function backoffDelay(attempt: number): number {
  const exp = Math.min(attempt, 10); // Cap exponent to prevent overflow
  const base = RETRY_BASE_DELAY_MS * Math.pow(2, exp);
  const jitter = Math.random() * base * 0.5;
  return base + jitter;
}

// ── Custom streaming with retry ─────────────────────────────────────────────

/**
 * Build tool instructions for Gemma or Qwen models and inject them
 * into the context's system prompt. Returns a copy of the context with
 * modified system prompt if tools are present and model family matches.
 */
function injectToolInstructions(
  context: Context,
  modelFamily: "gemma" | "qwen",
): Context {
  if (!context.tools || context.tools.length === 0) return context;

  const piTools = context.tools as Tool<any>[];
  // Convert Tool objects to the format expected by instruction builders
  const toolDescs = piTools.map((t) => ({
    type: "function" as const,
    function: {
      name: t.name,
      description: t.description,
      parameters: t.parameters as any,
    },
  }));

  const instructions =
    modelFamily === "qwen"
      ? buildQwenToolInstructions(toolDescs)
      : buildGemmaToolInstructions(toolDescs);

  // Check if instructions are already injected
  const existingCheck = "You have access to the following functions. To call a function, you MUST";
  if (context.systemPrompt && context.systemPrompt.includes(existingCheck)) {
    return context;
  }

  return {
    ...context,
    systemPrompt: (context.systemPrompt ?? "") + instructions,
    // Don't send tools in OpenAI format — the model uses text instructions
    tools: undefined,
  };
}

/**
 * Convert pi's internal message format to OpenAI Chat Completions format.
 *
 * For Gemma/Qwen (text-tool-call models), we collapse prior assistant
 * toolCall blocks and toolResult messages back into inline text using the
 * model's native format. Many local serving stacks (MLX, llama.cpp with
 * Gemma/Qwen templates) reject OpenAI `role: "tool"` and `tool_calls`
 * fields when `tools` isn't part of the request — which is exactly our
 * case for the text-injection path. Serializing tool history as plain
 * text lets the model see prior calls and results without the chat
 * template erroring out. This is the round-trip half of the
 * fixCocoreToolCalls path: text → structured toolCall blocks (output)
 * becomes structured toolCall blocks → text (history).
 */
function convertMessagesForOpenAI(
  messages: Message[],
  modelFamily?: "gemma" | "qwen",
): unknown[] {
  const result: unknown[] = [];

  for (let i = 0; i < messages.length; i++) {
    const msg = messages[i];

    if (msg.role === "user") {
      if (typeof msg.content === "string") {
        result.push({ role: "user", content: msg.content });
      } else {
        const parts = msg.content.map((c) => {
          if (c.type === "text") return { type: "text", text: c.text };
          return {
            type: "image_url",
            image_url: { url: `data:${c.mimeType};base64,${c.data}` },
          };
        });
        result.push({ role: "user", content: parts });
      }
    } else if (msg.role === "assistant") {
      if (modelFamily) {
        // Text-tool-call models: re-render tool calls inline so the chat
        // template never sees tool_calls. Output text stays in text blocks;
        // thinking blocks fold into <thinking>...</thinking> text.
        const segments: string[] = [];
        for (const block of msg.content) {
          if (block.type === "text") {
            segments.push((block as { text: string }).text);
          } else if (block.type === "thinking") {
            segments.push(
              `<thinking>${(block as { thinking: string }).thinking}</thinking>`,
            );
          } else if (block.type === "toolCall") {
            const tc = block as ToolCall;
            const args = JSON.stringify(tc.arguments ?? {});
            segments.push(
              modelFamily === "qwen"
                ? `\n<tool_call>\n${JSON.stringify({ name: tc.name, arguments: tc.arguments ?? {} })}\n</tool_call>\n`
                : `<|tool_call|>${tc.name}${args}`,
            );
          }
        }
        result.push({ role: "assistant", content: segments.join("") });
      } else {
        const content: unknown[] = [];
        const toolCalls: unknown[] = [];
        for (const block of msg.content) {
          if (block.type === "text") {
            content.push({ type: "text", text: block.text });
          } else if (block.type === "thinking") {
            content.push({ type: "text", text: `<thinking>${block.thinking}</thinking>` });
          } else if (block.type === "toolCall") {
            toolCalls.push({
              id: block.id,
              type: "function",
              function: {
                name: block.name,
                arguments: JSON.stringify(block.arguments),
              },
            });
          }
        }
        const assistantMsg: Record<string, unknown> = {
          role: "assistant",
          content: content.length > 0 ? content : null,
        };
        if (toolCalls.length > 0) {
          assistantMsg.tool_calls = toolCalls;
        }
        result.push(assistantMsg);
      }
    } else if (msg.role === "toolResult") {
      if (modelFamily) {
        // Text-tool-call models: surface the result as a user message
        // describing what the function returned. Preserves the
        // toolName so the model can correlate with the prior call.
        const text = msg.content
          .filter((c) => c.type === "text")
          .map((c) => (c as { text: string }).text)
          .join("\n");
        const toolName = (msg as { toolName?: string }).toolName ?? "tool";
        result.push({
          role: "user",
          content: `Function ${toolName} returned:\n${text}`,
        });
      } else {
        const content = msg.content
          .filter((c) => c.type === "text")
          .map((c) => c.text)
          .join("\n");
        result.push({
          role: "tool",
          tool_call_id: msg.toolCallId,
          content,
        });
      }
    }
  }

  return result;
}

/**
 * Per-request routing and policy fields. The extension passes these
 * through unchanged so the upstream API can shape provider selection.
 * `country` and `minProviderVersion` are optional on every tier;
 * `minTrust` is only meaningful on the verified route.
 */
interface RequestRoutingOptions {
  country?: string;
  minProviderVersion?: string;
  /** Routing tier used for this request — controls the URL path. */
  routing: CocoreRouting;
}

/**
 * Build the OpenAI-compatible request body sent to cocore.dev.
 *
 * Pulled out of `streamCocore` so the body shape — including the
 * reasoning-mode handling and routing fields — is testable without
 * spinning up a fetch mock. Three non-obvious bits:
 *
 *  1. `chat_template_kwargs: { enable_thinking: false }` is appended when
 *     pi passes `reasoning: "off"`. Without this, Qwen3 (and other
 *     reasoning-capable local models served through MLX) emit
 *     `<think>...</think>` inline in `delta.content`, which leaks the
 *     model's reasoning into the visible response.
 *
 *  2. `tools` is only sent in OpenAI format for non-Gemma/Qwen families;
 *     the text-tool-call path injects tool instructions into the system
 *     prompt and strips the array so the chat template doesn't choke on
 *     OpenAI-style `tool_calls` it can't render.
 *
 *  3. `country` / `min_provider_version` / `min_trust` are passed
 *     through verbatim when configured. The verified route gets a
 *     `min_trust: "hardware-attested"` default so unconfigured users
 *     land on the cryptographically-verified provider pool rather
 *     than the open route's self-asserted labels.
 */
function buildCocoreRequestBody(
  model: Model<Api>,
  effectiveContext: Context,
  family: "gemma" | "qwen" | null,
  options: SimpleStreamOptions | undefined,
  routingOpts: RequestRoutingOptions,
): Record<string, unknown> {
  const messages = convertMessagesForOpenAI(
    effectiveContext.messages,
    family ?? undefined,
  );
  if (effectiveContext.systemPrompt) {
    messages.unshift({
      role: "system",
      content: effectiveContext.systemPrompt,
    });
  }

  const body: Record<string, unknown> = {
    model: model.id,
    messages,
    stream: true,
    stream_options: { include_usage: true },
  };

  if (options?.maxTokens) {
    body.max_tokens = options.maxTokens;
  }
  if (options?.temperature !== undefined) {
    body.temperature = options.temperature;
  }

  // Qwen3 / Gemma thinking control. Other reasoning levels are deliberately
  // not mapped — there's no standardized mapping across MLX/Qwen serving
  // stacks for "low" / "medium" / etc., and we'd rather let the model use
  // its own defaults than guess wrong. Defensive stripping of leaked
  // thinking content happens at text_end (see `stripThinkingContent`).
  if (options?.reasoning === "off") {
    body.chat_template_kwargs = { enable_thinking: false };
  }

  // Only include tools in OpenAI format for non-Gemma/Qwen models
  if (!family && effectiveContext.tools && effectiveContext.tools.length > 0) {
    body.tools = effectiveContext.tools.map((t: Tool<any>) => ({
      type: "function",
      function: {
        name: t.name,
        description: t.description,
        parameters: t.parameters,
      },
    }));
  }

  // Routing / dispatch pins. ISO 3166-1 alpha-2 for country; the docs
  // accept a leading "v" on `min_provider_version`, so we strip it to
  // keep the parser on the server happy.
  if (routingOpts.country) {
    body.country = routingOpts.country;
  }
  if (routingOpts.minProviderVersion) {
    body.min_provider_version = routingOpts.minProviderVersion.replace(/^v/i, "");
  }
  if (routingOpts.routing === "verified") {
    body.min_trust = "hardware-attested";
  }

  return body;
}

/**
 * Custom streaming implementation for Co/Core provider with retry logic.
 *
 * Handlers:
 * - Injects tool-calling instructions for Gemma/Qwen models
 * - Retries on idle-timeout and transient server errors
 * - Parses OpenAI-compatible SSE stream
 */
/**
 * Dependencies that can be injected for testing. Production callers leave
 * `deps` undefined; the stream falls back to `globalThis.fetch`.
 *
 * `routing` selects which `/v1/<tier>/chat/completions` URL the request
 * hits. Defaults to "open" so tests that don't care about routing don't
 * need to pass it.
 *
 * `country` and `minProviderVersion` are forwarded into the request
 * body when set. They come from the saved cocore config at registration
 * time, not from per-call options — they're user preferences, not
 * per-request knobs.
 */
export interface StreamCocoreDeps {
  fetchImpl?: typeof fetch;
  routing?: CocoreRouting;
  country?: string;
  minProviderVersion?: string;
}

function streamCocore(
  model: Model<Api>,
  context: Context,
  options?: SimpleStreamOptions,
  deps?: StreamCocoreDeps,
): AssistantMessageEventStream {
  const stream = createAssistantMessageEventStream();
  const fetchImpl = deps?.fetchImpl ?? globalThis.fetch;
  const routing: CocoreRouting = deps?.routing ?? "open";

  (async () => {
    const maxRetries = options?.maxRetries ?? MAX_RETRIES;
    const signal = options?.signal;
    const apiKey = options?.apiKey;

    // Detect model family for tool instruction injection
    const family = getModelFamily(model.id);
    const effectiveContext = family
      ? injectToolInstructions(context, family)
      : context;

    const output: AssistantMessage = {
      role: "assistant",
      content: [],
      api: model.api,
      provider: model.provider,
      model: model.id,
      usage: {
        input: 0,
        output: 0,
        cacheRead: 0,
        cacheWrite: 0,
        totalTokens: 0,
        cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, total: 0 },
      },
      stopReason: "stop",
      timestamp: Date.now(),
    };

    let lastErrorMessage: string | undefined;
    let textContentIndex: number | null;
    const toolCallAccumulators: Map<
      number,
      { id: string; name: string; json: string; contentIdx: number }
    > = new Map();

    for (let attempt = 0; attempt <= maxRetries; attempt++) {
      if (signal?.aborted) {
        output.stopReason = "aborted";
        output.errorMessage = "Request was aborted";
        stream.push({ type: "error", reason: "aborted", error: output });
        stream.end();
        return;
      }

      if (attempt > 0) {
        const delay = backoffDelay(attempt - 1);
        console.log(
          `[cocore] Retry attempt ${attempt}/${maxRetries} after ${Math.round(delay)}ms (previous error: ${lastErrorMessage})`,
        );
        await sleep(delay);

        // Re-check abort after waiting
        if (signal?.aborted) {
          output.stopReason = "aborted";
          output.errorMessage = "Request was aborted";
          stream.push({ type: "error", reason: "aborted", error: output });
          stream.end();
          return;
        }
      }

      try {
        // Reset output content for this attempt (in case of retry)
        output.content = [];
        output.usage = {
          input: 0,
          output: 0,
          cacheRead: 0,
          cacheWrite: 0,
          totalTokens: 0,
          cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, total: 0 },
        };
        output.stopReason = "stop";

        // Build request payload
        const body = buildCocoreRequestBody(
          model,
          effectiveContext,
          family,
          options,
          {
            routing,
            country: deps?.country,
            minProviderVersion: deps?.minProviderVersion,
          },
        );

        console.log(`[cocore:${routing}] sending request (attempt ${attempt + 1}/${maxRetries + 1})`);

        const headers: Record<string, string> = {
          "Content-Type": "application/json",
          Authorization: `Bearer ${apiKey}`,
        };

        // Merge model-level and provider-level headers
        if (model.headers) Object.assign(headers, model.headers);
        if (options?.headers) Object.assign(headers, options.headers);

        const fetchTimeout = options?.timeoutMs ?? 600_000; // Default 10 min
        const controller = new AbortController();
        const timeoutId = setTimeout(() => controller.abort(), fetchTimeout);

        // Link to parent abort signal if provided
        const onAbort = () => controller.abort();
        signal?.addEventListener("abort", onAbort, { once: true });

        let response: Response;
        try {
          response = await fetchImpl(
            `${model.baseUrl || BASE_URL}${chatCompletionsPath(routing)}`,
            {
              method: "POST",
              headers,
              body: JSON.stringify(body),
              signal: controller.signal,
            },
          );
        } finally {
          clearTimeout(timeoutId);
          signal?.removeEventListener("abort", onAbort);
        }

        if (!response.ok) {
          const errorText = await response.text().catch(() => "");
          const parsed = parseServerError(response.status, errorText, model.id);
          lastErrorMessage = parsed.friendly;

          if (parsed.retryable && attempt < maxRetries) {
            console.log(
              `[cocore:${routing}] ${parsed.code ?? `HTTP ${response.status}`}: ${errorText.slice(0, 200)}. Will retry.`,
            );
            continue; // Retry
          }

          // Not retryable, or exhausted retries
          output.stopReason = "error";
          output.errorMessage = parsed.friendly;
          stream.push({ type: "error", reason: "error", error: output });
          stream.end();
          return;
        }

        // Success — parse the SSE stream
        stream.push({ type: "start", partial: output });

        const reader = response.body?.getReader();
        if (!reader) {
          throw new Error("Response body is not readable");
        }

        const decoder = new TextDecoder();
        let buffer = "";
        textContentIndex = null;
        toolCallAccumulators.clear();

        try {
          while (true) {
            const { done, value } = await reader.read();
            if (done) break;

            buffer += decoder.decode(value, { stream: true });

            // Process complete SSE lines
            const lines = buffer.split("\n");
            // Keep the last (potentially incomplete) line in the buffer
            buffer = lines.pop() || "";

            for (const line of lines) {
              if (!line.startsWith("data: ")) continue;
              const data = line.slice(6).trim();
              if (data === "[DONE]") continue;

              let chunk: Record<string, unknown>;
              try {
                chunk = JSON.parse(data);
              } catch {
                continue;
              }

              const choices = chunk.choices as
                | Array<Record<string, unknown>>
                | undefined;
              if (!choices || choices.length === 0) continue;
              const choice = choices[0];
              const delta = choice.delta as Record<string, unknown> | undefined;

              // Handle usage info
              if (chunk.usage) {
                const u = chunk.usage as Record<string, unknown>;
                output.usage.input = (u.prompt_tokens as number) ?? 0;
                output.usage.output = (u.completion_tokens as number) ?? 0;
                const details = u.prompt_tokens_details as Record<string, number> | undefined;
                output.usage.cacheRead = details?.cached_tokens ?? 0;
                output.usage.totalTokens =
                  output.usage.input +
                  output.usage.output +
                  output.usage.cacheRead +
                  output.usage.cacheWrite;
                calculateCost(model, output.usage as Usage);
              }

              // Handle finish reason
              if (choice.finish_reason) {
                const reason = choice.finish_reason as string;
                if (reason === "tool_calls") {
                  output.stopReason = "toolUse";
                } else if (reason === "length") {
                  output.stopReason = "length";
                } else if (reason === "stop") {
                  output.stopReason = "stop";
                }
              }

              if (!delta || Object.keys(delta as Record<string, unknown>).length === 0) continue;

              // Text content
              if (delta.content) {
                if (textContentIndex === null) {
                  textContentIndex = output.content.length;
                  output.content.push({ type: "text", text: "" });
                  stream.push({
                    type: "text_start",
                    contentIndex: textContentIndex,
                    partial: output,
                  });
                }
                const block = output.content[textContentIndex];
                if (block.type === "text") {
                  block.text += delta.content as string;
                  stream.push({
                    type: "text_delta",
                    contentIndex: textContentIndex,
                    delta: delta.content as string,
                    partial: output,
                  });
                }
              }

              // Tool calls
              const toolCalls = delta.tool_calls as
                | Array<Record<string, unknown>>
                | undefined;
              if (toolCalls) {
                for (const tc of toolCalls) {
                  const idx = tc.index as number;

                  let accum = toolCallAccumulators.get(idx);
                  if (!accum) {
                    const contentIdx = output.content.length;
                    accum = {
                      id: (tc.id as string) || "",
                      name: "",
                      json: "",
                      contentIdx,
                    };
                    toolCallAccumulators.set(idx, accum);

                    // Add placeholder to output content
                    output.content.push({
                      type: "toolCall",
                      id: accum.id,
                      name: "",
                      arguments: {},
                    });
                    stream.push({
                      type: "toolcall_start",
                      contentIndex: contentIdx,
                      partial: output,
                    });
                  }

                  if (tc.id) accum.id = tc.id as string;

                  const fn = tc.function as Record<string, unknown> | undefined;
                  if (fn) {
                    if (fn.name) accum.name = fn.name as string;
                    if (fn.arguments) accum.json += fn.arguments as string;
                  }

                  const block = output.content[accum.contentIdx];
                  if (block && block.type === "toolCall") {
                    block.id = accum.id;
                    block.name = accum.name;
                    try {
                      block.arguments = JSON.parse(accum.json);
                    } catch {
                      // Partial JSON — keep previous parse result
                    }
                    stream.push({
                      type: "toolcall_delta",
                      contentIndex: accum.contentIdx,
                      delta: (fn?.arguments as string) ?? "",
                      partial: output,
                    });
                  }
                }
              }
            }
          }
        } finally {
          reader.releaseLock();
        }

        // End text block if one was started
        if (textContentIndex !== null) {
          const block = output.content[textContentIndex];
          if (block && block.type === "text") {
            // Defensive: strip any <think>...</think> ranges that leaked
            // through despite reasoning: "off" being sent. Without this,
            // a model whose chat template ignores enable_thinking would
            // show its reasoning to the user.
            const cleaned = stripThinkingContent(block.text);
            if (cleaned !== block.text) {
              block.text = cleaned;
            }
            stream.push({
              type: "text_end",
              contentIndex: textContentIndex,
              content: block.text,
              partial: output,
            });
          }
        }

        // End any tool call blocks
        for (const [_idx, accum] of toolCallAccumulators) {
          const block = output.content[accum.contentIdx];
          if (block && block.type === "toolCall") {
            try {
              block.arguments = JSON.parse(accum.json);
            } catch {
              // Keep whatever was parsed
            }
            stream.push({
              type: "toolcall_end",
              contentIndex: accum.contentIdx,
              toolCall: block as ToolCall,
              partial: output,
            });
          }
        }

        // Success path — but first, gate against empty responses.
        const hasContent =
          output.content.length > 0 ||
          output.usage.totalTokens > 0;

        if (!hasContent && attempt < maxRetries) {
          // Empty response — likely idle-timeout on the server
          lastErrorMessage = "empty response (likely idle-timeout)";
          console.log(
            `[cocore:${routing}] Empty response received (0 content, 0 tokens). Will retry.`,
          );
          continue; // Retry
        }

        if (!hasContent) {
          // Exhausted retries (or single attempt with maxRetries=0) and the
          // server still returned nothing usable. Surface this as an error
          // event so pi's UI shows the user something actionable instead of
          // silently emitting a `done` with an empty message.
          output.stopReason = "error";
          output.errorMessage =
            lastErrorMessage ?? "Empty response from Co/Core after retries";
          console.error(
            `[cocore:${routing}] Empty response after ${attempt + 1} attempt(s); surfacing error to UI.`,
          );
          stream.push({ type: "error", reason: "error", error: output });
          stream.end();
          return;
        }

        stream.push({
          type: "done",
          reason: output.stopReason as Extract<StopReason, "stop" | "length" | "toolUse">,
          message: output,
        });
        stream.end();
        return;
      } catch (err) {
        lastErrorMessage = err instanceof Error ? err.message : String(err);

        if (signal?.aborted) {
          output.stopReason = "aborted";
          output.errorMessage = "Request was aborted";
          stream.push({ type: "error", reason: "aborted", error: output });
          stream.end();
          return;
        }

        if (attempt < maxRetries) {
          console.log(
            `[cocore:${routing}] Request error: ${lastErrorMessage}. Will retry.`,
          );
          continue; // Retry on network errors
        }

        // Exhausted retries
        output.stopReason = "error";
        output.errorMessage = lastErrorMessage;
        stream.push({ type: "error", reason: "error", error: output });
        stream.end();
        return;
      }
    }

    // Should not reach here, but handle it
    output.stopReason = "error";
    output.errorMessage = lastErrorMessage ?? "Unknown error after retries";
    stream.push({ type: "error", reason: "error", error: output });
    stream.end();
  })();

  return stream;
}

// ── Model detection ──────────────────────────────────────────────────────────

/**
 * Check if a model is served through the Co/Core provider.
 */
function isCocoreModel(model: { model?: string; provider?: string }): boolean {
  return ROUTINGS.some((routing) => model.provider === providerKey(routing));
}

/**
 * Check if a model is a Gemma variant (3 or 4) from Co/Core.
 */
function isGemmaModel(model: { model?: string; provider?: string }): boolean {
  return isCocoreModel(model) && /\bgemma\b/i.test(model.model ?? "");
}

/**
 * Check if a model is a Qwen variant (2.5 or 3) from Co/Core.
 */
function isQwenModel(model: { model?: string; provider?: string }): boolean {
  return isCocoreModel(model) && /\bqwen\b/i.test(model.model ?? "");
}

// ── Tool instruction builders ────────────────────────────────────────────────

/** Extract a JSON substring starting at openBraceIdx using brace counting. */
function extractJsonFrom(text: string, openBraceIdx: number): string | null {
  if (text[openBraceIdx] !== "{") return null;
  let depth = 0;
  let inString = false;
  let escape = false;
  for (let i = openBraceIdx; i < text.length; i++) {
    const ch = text[i];
    if (escape) {
      escape = false;
      continue;
    }
    if (ch === "\\") {
      escape = true;
      continue;
    }
    if (ch === '"') {
      inString = !inString;
      continue;
    }
    if (inString) continue;
    if (ch === "{") depth++;
    if (ch === "}") {
      depth--;
      if (depth === 0) {
        return text.slice(openBraceIdx, i + 1);
      }
    }
  }
  return null;
}

interface ToolCallMatch {
  start: number;
  end: number;
  name: string;
  arguments: Record<string, unknown>;
}

/**
 * Build tool-calling instructions for Qwen models.
 * Qwen 2.5/3 use: <tool_call>\n{"name":"func","arguments":{...}}\n</tool_call>
 */
function buildQwenToolInstructions(tools: Array<{ type: string; function: { name: string; description?: string; parameters?: { type: string; properties?: Record<string, { type?: string; description?: string }>; required?: string[] } } }>): string {
  let text = "\n\n# Tools\n\n";
  text += "You have access to the following functions. To call a function, you MUST output ONLY a JSON block in this exact format:\n\n";
  text += "<tool_call>\n";
  text += '{"name": "function_name", "arguments": {"param1": "value1"}}\n';
  text += "</tool_call>\n\n";
  text += "Available functions:\n";
  for (const tool of tools) {
    const func = tool.function;
    text += `\n### ${func.name}\n`;
    text += `${func.description || "No description"}\n`;
    if (func.parameters?.properties) {
      const required = func.parameters.required || [];
      for (const [pname, pdef] of Object.entries(func.parameters.properties)) {
        const req = required.includes(pname) ? " (required)" : "";
        text += `  - ${pname}${req}: ${pdef.description || pdef.type || "any"}\n`;
      }
    }
  }
  text += "\nWhen you need to use a tool, output ONLY the <tool_call> block. Do not include any other text, explanations, or code. The tool result will be provided to you.\n";
  return text;
}

/**
 * Build tool-calling instructions for Gemma models.
 * Gemma 3: <|tool_call|>func{json}<|tool_call|>func2{json}
 * Gemma 4: <|tool_call>call:func{json}<|tool_call|>
 *
 * We use the Gemma 3 format since it's simpler and works for both variants.
 */
function buildGemmaToolInstructions(tools: Array<{ type: string; function: { name: string; description?: string; parameters?: { type: string; properties?: Record<string, { type?: string; description?: string }>; required?: string[] } } }>): string {
  let text = "\n\n# Tools\n\n";
  text += "You have access to the following functions. To call a function, you MUST use this exact format:\n\n";
  text += '<|tool_call|>function_name{"param1": "value1"}\n\n';
  text += "For multiple function calls, separate each with <|tool_call|>:\n\n";
  text += '<|tool_call|>first_func{"param":"value"}<|tool_call|>second_func{"param":"value"}\n\n';
  text += "Available functions:\n";
  for (const tool of tools) {
    const func = tool.function;
    text += `\n### ${func.name}\n`;
    text += `${func.description || "No description"}\n`;
    if (func.parameters?.properties) {
      const required = func.parameters.required || [];
      for (const [pname, pdef] of Object.entries(func.parameters.properties)) {
        const req = required.includes(pname) ? " (required)" : "";
        text += `  - ${pname}${req}: ${pdef.description || pdef.type || "any"}\n`;
      }
    }
  }
  text += "\nWhen you need to use a tool, output ONLY the tool call. Do not include any other text, explanations, or code.\n";
  return text;
}

// ── Tool call parsers ────────────────────────────────────────────────────────

/**
 * Strip leaked `<think>...</think>` ranges from model output text.
 *
 * Defensive backstop for the reasoning-mode flag in
 * `buildCocoreRequestBody`: when pi sends `reasoning: "off"`, the chat
 * template should suppress thinking tokens at the source, but local
 * serving stacks (MLX, llama.cpp) don't always honor `enable_thinking`
 * consistently. The model then emits the reasoning block inline in
 * `delta.content`, which would otherwise reach the user.
 *
 * Tolerates both `</think>` and `</think>` close variants — different
 * chat templates use different tokens. An unterminated `<think>` is
 * stripped to end of text (the model failed to close; better to drop
 * the dangling fragment than to leak it).
 *
 * Also handles the orphan-close case: some Qwen3 variants start
 * reasoning at the very beginning of `delta.content` without emitting a
 * matching `<think>` opener, and only emit the close tag. A naïve regex
 * that requires `<think>` would miss this and leak the whole reasoning
 * block — captured verbatim from a `mlx-community/Qwen3.5-4B-MLX-4bit`
 * session after PR #3 landed. We detect orphan closes by comparing the
 * position of the first close tag against the first opener: if the
 * close appears before any opener (or no opener exists), treat the
 * whole prefix as thinking.
 */
function stripThinkingContent(text: string): string {
  // Step 1: paired <think>...</think> ranges (and orphan-opens stripped
  // to end of text via the `$` fallback in the close alternation).
  const pairedStripped = text.replace(
    /<think>[\s\S]*?(?:<\/?think>|$)/g,
    "",
  );

  // Step 2: orphan close. Inspect the ORIGINAL text to decide whether
  // the reasoning started at position 0 (no opener seen before the
  // first close tag). If so, strip from start to the first close. This
  // is independent of step 1 because paired ranges may have already
  // been removed, but the orphan-close diagnosis comes from the raw
  // input where we can see the close appeared without a preceding open.
  const firstOpenIdx = text.indexOf("<think>");
  const firstCloseMatch = text.match(/<\/?think>/);
  if (
    firstCloseMatch !== null &&
    (firstOpenIdx === -1 || firstOpenIdx > firstCloseMatch.index)
  ) {
    return text
      .slice(firstCloseMatch.index + firstCloseMatch[0].length)
      .trim();
  }

  return pairedStripped.trim();
}

/**
 * Strip chat-template control-token escapes that some local serving
 * stacks (MLX, llama.cpp with Gemma/Qwen chat templates) leak into
 * the model's text output instead of decoding them back to the
 * underlying character. Without this, the parser downstream can't make
 * sense of the surrounding JSON.
 *
 * Conservative: only known control-token escapes are replaced. Anything
 * unrecognized passes through unchanged.
 *
 * The literal-quote escape (`<|"|>`) is the load-bearing one: it lets
 * the model emit a JSON string value that itself contains `"` chars
 * (e.g. a bash command with `format:"%h %s"` in it). Mapping it to
 * `\"` keeps the surrounding JSON parseable.
 */
function normalizeModelText(text: string): string {
  // Qwen / Gemma literal-quote escape — model means `\"` inside a JSON
  // value. Empirical: a bash command like `format:"%h %s"` is wrapped
  // in `<|"|>` so the inner `"` doesn't close the surrounding JSON
  // string early. Add more escapes here as they show up in the wild;
  // keep the list small and well-documented so unrelated text isn't
  // silently rewritten.
  return text.replace(/<\|"\|>/g, '\\"');
}

/**
 * Parse Gemma 3 tool calls from text.
 *
 * Gemma 3 format: <|tool_call|>func{json}<|tool_call|>func2{json2}
 * `<|tool_call|>` acts as a delimiter. Each segment between delimiters
 * is a function call: `func_name{json_args}`.
 */
function parseGemma3ToolCalls(text: string): ToolCallMatch[] {
  const results: ToolCallMatch[] = [];
  const delim = "<|tool_call|>";
  let searchFrom = 0;

  while (searchFrom < text.length) {
    const delimIdx = text.indexOf(delim, searchFrom);
    if (delimIdx === -1) break;

    // Content starts after the delimiter
    const contentStart = delimIdx + delim.length;

    // Find the next delimiter or end of text
    const nextDelim = text.indexOf(delim, contentStart);
    const contentEnd = nextDelim === -1 ? text.length : nextDelim;

    const segment = text.slice(contentStart, contentEnd).trim();
    if (segment.length > 0) {
      // Parse func_name{json} from segment
      const braceIdx = segment.indexOf("{");
      if (braceIdx > 0) {
        const funcName = segment.slice(0, braceIdx).trim();
        const jsonStr = extractJsonFrom(segment, braceIdx);
        if (jsonStr && funcName) {
          try {
            const args = JSON.parse(jsonStr);
            results.push({
              start: delimIdx,
              end: contentEnd,
              name: funcName,
              arguments: args,
            });
          } catch {
            // Malformed JSON — skip
          }
        }
      }
    }

    searchFrom = nextDelim === -1 ? text.length : nextDelim;
  }

  return results;
}

/**
 * Parse Gemma 4 tool calls from text.
 *
 * Gemma 4 nominal format:
 *   <|tool_call>call:func_name{json}</​tool_call​>
 *
 * The opening tag ends in `>`; the closing tag ends in `|>`. The model
 * doesn't always honour that asymmetry — observed variants:
 *
 *   - Open and close swapped: model emits `<|tool_call|>call:…​<tool_call|>`
 *     (the close-marker used as the opener; opener-marker or a mangled
 *     fragment used as the close).
 *   - Control-token escapes (`<|"|>`) leak through the tokenizer into
 *     the body, leaving the embedded JSON unparseable until stripped.
 *
 * Strategy: tolerate all four delimiter variants at both ends, then let
 * the existing JSON-extraction path handle the body. Caller is expected
 * to have run `normalizeModelText` on the input (parseToolCalls does
 * this once at the top so direct callers get the same treatment).
 */
const GEMMA4_TOOL_CALL_RE =
  /<\|?tool_call\|?>(?:call:)?([a-zA-Z0-9_]+)\s*(\{[\s\S]+?\})<\|?tool_call\|?>/g;

function parseGemma4ToolCalls(text: string): ToolCallMatch[] {
  const results: ToolCallMatch[] = [];
  let match;
  GEMMA4_TOOL_CALL_RE.lastIndex = 0;
  while ((match = GEMMA4_TOOL_CALL_RE.exec(text)) !== null) {
    const name = match[1];
    const argsStr = match[2];
    // Lazy quantifier stops at the first `}`. extractJsonFrom balances
    // braces properly so nested objects inside `arguments` survive.
    const braceIdx = argsStr.indexOf("{");
    const fullJson = braceIdx >= 0 ? extractJsonFrom(argsStr, braceIdx) : argsStr;
    try {
      const args = JSON.parse(fullJson ?? argsStr);
      results.push({
        start: match.index,
        end: GEMMA4_TOOL_CALL_RE.lastIndex,
        name,
        arguments: args,
      });
    } catch {
      // JSON.parse failed. The model often emits a pseudo-JSON shape
      // like {key:\"value\"} — unquoted property name, value bounded
      // by the model's literal-quote escape, and raw `"` chars inside
      // the value that never got escaped. Fall back to a single-key
      // splitter that handles that shape.
      const pseudo = parseGemmaPseudoJson(argsStr);
      if (pseudo) {
        results.push({
          start: match.index,
          end: GEMMA4_TOOL_CALL_RE.lastIndex,
          name,
          arguments: pseudo,
        });
      }
    }
  }
  return results;
}

/**
 * Parse Gemma's pseudo-JSON tool-call body. After normalizeModelText
 * strips the `<|"|>` literal-quote escape, the model often emits:
 *
 *   {key:\"value\"}
 *
 * — single key-value pair, unquoted property name, value wrapped in
 * escape-quotes. Real JSON.parse rejects this on two counts: the
 * unquoted key, and any bare `"` chars inside the value that the
 * model didn't bother to escape. This splitter handles the single-
 * key case (the dominant one in practice). Multi-key and nested
 * shapes still require the upstream model/template to emit real JSON.
 *
 * Returns null if the shape doesn't match. The dispatcher decides
 * what to do with that.
 */
function parseGemmaPseudoJson(body: string): Record<string, string> | null {
  const trimmed = body.trim();
  const inner =
    trimmed.startsWith("{") && trimmed.endsWith("}")
      ? trimmed.slice(1, -1).trim()
      : trimmed;
  if (!inner) return null;

  // Split on the FIRST `:` to separate key from value. Values may
  // contain additional colons (URLs, time formats, shell `--flag:val`),
  // so we don't split on every colon.
  const colonIdx = inner.indexOf(":");
  if (colonIdx <= 0) return null;

  const rawKey = inner.slice(0, colonIdx).trim();
  let rawValue = inner.slice(colonIdx + 1).trim();
  if (!rawKey || !rawValue) return null;

  // Strip the surrounding escape-quoted pair (\"...\") or regular
  // quote pair ("...") from the value if both ends are present.
  if (
    rawValue.length >= 4 &&
    rawValue.startsWith('\\"') &&
    rawValue.endsWith('\\"')
  ) {
    rawValue = rawValue.slice(2, -2);
  } else if (rawValue.length >= 2 && rawValue.startsWith('"') && rawValue.endsWith('"')) {
    rawValue = rawValue.slice(1, -1);
  }

  return { [rawKey]: rawValue };
}

/**
 * Last-resort extractor for Gemma-shaped tool calls when the strict
 * `<|tool_call|>…​<|/tool_call|>` envelope is missing or mangled beyond
 * what the tolerant regex can handle. Scans for `func_name{json}` pairs
 * separated by reasonable whitespace.
 *
 * Used as a third pass in the gemma path. Should not be reached for
 * well-formed Gemma 4 output; if you see it firing in normal use,
 * the upstream model / template has drifted further and needs a
 * dedicated parser.
 */
function parsePositionalGemmaCalls(text: string): ToolCallMatch[] {
  const results: ToolCallMatch[] = [];
  // `name{...}` followed by another `name{...}` or end of string.
  // Conservative: names must be a single identifier; bodies must start
  // with `{`. Both delimiters (if present) are consumed by the caller
  // when applicable — this function operates on raw-ish text.
  const re = /(^|[\s>])([a-zA-Z][a-zA-Z0-9_]*)\s*(\{[\s\S]+?\})(?=\s*[a-zA-Z][a-zA-Z0-9_]*\s*\{|$)/g;
  let match;
  while ((match = re.exec(text)) !== null) {
    const name = match[2];
    const argsStr = match[3];
    const braceIdx = argsStr.indexOf("{");
    const fullJson = braceIdx >= 0 ? extractJsonFrom(argsStr, braceIdx) : argsStr;
    try {
      const args = JSON.parse(fullJson ?? argsStr);
      results.push({
        start: match.index + match[1].length,
        end: match.index + match[0].length,
        name,
        arguments: args,
      });
    } catch {
      // Skip — the JSON truly is malformed at this point.
    }
  }
  return results;
}

/**
 * Parse Qwen tool calls from text.
 *
 * Qwen format:
 *   1. <tool_call>\n{"name":"func","arguments":{...}}\n</tool_call>
 *   2. Bare {"name":"func","arguments":{...}} (one per line or separated by blank lines)
 *   3. Multiple blocks separated by newlines
 */
function parseQwenToolCalls(text: string): ToolCallMatch[] {
  const results: ToolCallMatch[] = [];

  // Strategy 1: Look for <tool_call>...</tool_call> blocks
  const blockRe = /<tool_call>\s*(\{[\s\S]*?\})\s*<\/tool_call>/g;
  let match;
  while ((match = blockRe.exec(text)) !== null) {
    const jsonStr = match[1];
    try {
      const parsed = JSON.parse(jsonStr);
      if (parsed && typeof parsed.name === "string" && typeof parsed.arguments === "object") {
        results.push({
          start: match.index,
          end: blockRe.lastIndex,
          name: parsed.name,
          arguments: parsed.arguments,
        });
      }
    } catch {
      // Malformed JSON — skip
    }
  }

  if (results.length > 0) return results;

  // Strategy 2: Find bare {"name":"...","arguments":{...}} objects
  // Search for JSON objects that look like tool calls
  const bareRe = /\{\s*"name"\s*:\s*"([a-zA-Z_][a-zA-Z0-9_]*)"\s*,\s*"arguments"\s*:/g;
  while ((match = bareRe.exec(text)) !== null) {
    const jsonStr = extractJsonFrom(text, match.index);
    if (jsonStr) {
      try {
        const parsed = JSON.parse(jsonStr);
        if (parsed && typeof parsed.name === "string" && typeof parsed.arguments === "object") {
          results.push({
            start: match.index,
            end: match.index + jsonStr.length,
            name: parsed.name,
            arguments: parsed.arguments,
          });
          // Skip past this match for the next search
          bareRe.lastIndex = match.index + jsonStr.length;
        }
      } catch {
        // Malformed JSON — skip
      }
    }
  }

  return results;
}

/**
 * Parse tool calls from text for a given model family.
 *
 * Normalizes the input once (strips control-token escapes) so every
 * downstream parser sees clean text. Sub-parsers can therefore assume
 * `<|"|>` etc. have already been converted to `\"`.
 */
function parseToolCalls(
  text: string,
  modelFamily: "gemma" | "qwen",
): ToolCallMatch[] {
  const normalized = normalizeModelText(text);
  if (modelFamily === "gemma") {
    // Try Gemma 4 format first (most specific), fall back to Gemma 3,
    // then to the positional extractor as a last resort.
    const g4 = parseGemma4ToolCalls(normalized);
    if (g4.length > 0) return g4;
    const g3 = parseGemma3ToolCalls(normalized);
    if (g3.length > 0) return g3;
    return parsePositionalGemmaCalls(normalized);
  }
  if (modelFamily === "qwen") {
    return parseQwenToolCalls(normalized);
  }
  return [];
}

// ── Message fixer ────────────────────────────────────────────────────────────

/**
 * Post-process an assistant message to extract model-native tool calls
 * from text content and convert them to structured toolCall blocks.
 */
function fixCocoreToolCalls(
  message: { role: string; content: Array<{ type: string; text?: string }> },
  modelFamily: "gemma" | "qwen",
): { role: string; content: Array<{ type: string; text?: string }> } {
  if (message.role !== "assistant") return message;

  // Collect all text blocks and their positions
  const textBlocks: Array<{ index: number; text: string }> = [];
  for (let i = 0; i < message.content.length; i++) {
    const block = message.content[i];
    if (block.type === "text" && typeof block.text === "string") {
      textBlocks.push({ index: i, text: block.text });
    }
  }

  if (textBlocks.length === 0) return message;

  // Check all text blocks for tool calls
  let hasToolCalls = false;
  const allMatches: Array<{
    textIndex: number;
    matchStart: number;
    matchEnd: number;
    name: string;
    arguments: Record<string, unknown>;
  }> = [];

  for (const tb of textBlocks) {
    const matches = parseToolCalls(tb.text, modelFamily);
    if (matches.length > 0) {
      hasToolCalls = true;
      for (const m of matches) {
        allMatches.push({
          textIndex: tb.index,
          matchStart: m.start,
          matchEnd: m.end,
          name: m.name,
          arguments: m.arguments,
        });
      }
    }
  }

  if (!hasToolCalls) return message;

  // Build new content: split text blocks around tool-call regions, insert ToolCall blocks
  const newContent: Array<
    { type: string; text?: string } | { type: "toolCall"; id: string; name: string; arguments: Record<string, unknown> }
  > = [];

  for (let i = 0; i < message.content.length; i++) {
    const block = message.content[i];
    const blockMatches = allMatches.filter((m) => m.textIndex === i);

    if (block.type === "text" && typeof block.text === "string" && blockMatches.length > 0) {
      const sorted = [...blockMatches].sort((a, b) => a.matchStart - b.matchStart);

      let lastEnd = 0;
      for (const m of sorted) {
        // Text before this match
        if (m.matchStart > lastEnd) {
          const prefix = block.text.slice(lastEnd, m.matchStart);
          if (prefix.length > 0) {
            newContent.push({ type: "text", text: prefix });
          }
        }
        // Insert tool call block
        const toolCallId = `cocore-tc-${m.name}-${Date.now()}-${Math.random().toString(36).slice(2, 8)}`;
        newContent.push({
          type: "toolCall",
          id: toolCallId,
          name: m.name,
          arguments: m.arguments,
        });
        lastEnd = m.matchEnd;
      }

      // Text after the last match
      if (lastEnd < block.text.length) {
        const suffix = block.text.slice(lastEnd);
        if (suffix.length > 0) {
          newContent.push({ type: "text", text: suffix });
        }
      }
    } else {
      newContent.push(block);
    }
  }

  return { ...message, content: newContent };
}

// ── Types ────────────────────────────────────────────────────────────────────

/**
 * Persisted user config. `apiKey` is required; the other fields are
 * optional user preferences. Old configs that lack the new fields load
 * fine because each consumer treats them as unset.
 */
interface CocoreConfig {
  apiKey: string;
  /** ISO 3166-1 alpha-2 country code (e.g. "US"). */
  country?: string;
  /** Minimum tray-provider release (e.g. "0.9.32"). */
  minProviderVersion?: string;
}

interface CocoreModelEntry {
  id: string;
  object?: string;
  created?: number;
  owned_by?: string;
}

interface CocoreModelsResponse {
  object: string;
  data: CocoreModelEntry[];
}

interface ModelCapabilities {
  reasoning: boolean;
  input: ["text"];
  cost: { input: number; output: number; cacheRead: number; cacheWrite: number };
  contextWindow: number;
  maxTokens: number;
}

// ── Config persistence ───────────────────────────────────────────────────────

function loadConfig(): CocoreConfig | null {
  try {
    if (existsSync(CONFIG_PATH)) {
      const raw = readFileSync(CONFIG_PATH, "utf-8");
      return JSON.parse(raw) as CocoreConfig;
    }
  } catch {
    // Corrupt or missing — treat as unconfigured
  }
  return null;
}

function saveConfig(config: CocoreConfig): void {
  mkdirSync(CONFIG_DIR, { recursive: true });
  writeFileSync(CONFIG_PATH, JSON.stringify(config, null, 2), "utf-8");
}

function resolveOptionalConfigInput(
  input: string | undefined,
  current: string | undefined,
): string | undefined {
  const value = input?.trim();
  return value === "-" ? undefined : value || current;
}

// ── Model capability derivation ──────────────────────────────────────────────

/**
 * Derive model capabilities from the model ID since the /models endpoint
 * only returns minimal metadata (id, object, created, owned_by).
 *
 * Recognized families fall back to sensible defaults for unknown models.
 */
function deriveModelCapabilities(modelId: string): ModelCapabilities {
  const id = modelId.toLowerCase();

  // Default: conservative values for a modern small-to-mid LLM
  let contextWindow = 32_768;
  let maxTokens = 8_192;

  // ── Qwen 2.5 family ──────────────────────────────────────────────────
  if (id.includes("qwen2.5")) {
    // Smaller Qwen2.5 variants (0.5B, 1.5B, 3B): 32K context
    if (id.includes("0.5b") || id.includes("1.5b") || id.includes("3b")) {
      contextWindow = 32_768;
    } else {
      // 7B, 14B, 32B, 72B: 128K context
      contextWindow = 128_000;
    }
    maxTokens = 8_192;
  }

  // ── Qwen 3.5 / 3.6 family ───────────────────────────────────────────
  // Branches ahead of the `qwen3` catch-all so the newer sub-families
  // can be tuned independently. As of writing the catalog ships
  // mlx-community/Qwen3.5-{0.8B,4B,9B}-MLX-4bit; treat them the same
  // as Qwen 3 until upstream publishes a model card that says otherwise.
  else if (id.includes("qwen3.5") || id.includes("qwen3.6")) {
    contextWindow = 128_000;
    maxTokens = 8_192;
  }

  // ── Qwen 3 family ────────────────────────────────────────────────────
  else if (id.includes("qwen3")) {
    contextWindow = 128_000;
    maxTokens = 8_192;
  }

  // ── Gemma family ─────────────────────────────────────────────────────
  else if (id.includes("gemma-3")) {
    contextWindow = 32_768;
    maxTokens = 8_192;
  } else if (id.includes("gemma-4")) {
    contextWindow = 128_000;
    maxTokens = 8_192;
  }

  // ── Llama 3 family ───────────────────────────────────────────────────
  else if (
    id.includes("llama-3.3") ||
    id.includes("llama-3.2") ||
    id.includes("llama-3.1")
  ) {
    contextWindow = 128_000;
    maxTokens = 16_384;
  } else if (id.includes("llama-4")) {
    contextWindow = 128_000;
    maxTokens = 16_384;
  } else if (id.includes("llama")) {
    // Catch-all for future Llama releases
    contextWindow = 128_000;
    maxTokens = 8_192;
  }

  // ── Mistral / Mixtral ────────────────────────────────────────────────
  else if (id.includes("mistral") || id.includes("mixtral")) {
    contextWindow = 32_768;
    maxTokens = 8_192;
    if (id.includes("large") || id.includes("8x")) {
      contextWindow = 128_000;
    }
  }

  // ── DeepSeek ─────────────────────────────────────────────────────────
  else if (id.includes("deepseek")) {
    contextWindow = 128_000;
    maxTokens = 8_192;
    if (id.includes("r1")) {
      // R1 is a reasoning model
      return {
        reasoning: true,
        input: ["text"],
        cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 },
        contextWindow,
        maxTokens,
      };
    }
  }

  // ── Phi ──────────────────────────────────────────────────────────────
  else if (id.includes("phi-4") || id.includes("phi-3")) {
    contextWindow = 128_000;
    maxTokens = 4_096;
    if (id.includes("vision") || id.includes("multimodal")) {
      contextWindow = 128_000;
      maxTokens = 4_096;
    }
  }

  return {
    reasoning: false,
    input: ["text"],
    cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 },
    contextWindow,
    maxTokens,
  };
}

// ── Provider registration ────────────────────────────────────────────────────

/**
 * Build a streamSimple that bakes the routing tier and user-set
 * routing policy into the call. Each routing tier registers its own
 * pi provider entry, so users can pick a route from the model picker
 * rather than re-running /cocore-setup. country and minProviderVersion
 * are session-level preferences that apply across every tier.
 */
function makeStreamForRouting(
  routing: CocoreRouting,
  routingPrefs: { country?: string; minProviderVersion?: string },
) {
  return (
    model: Model<Api>,
    context: Context,
    options?: SimpleStreamOptions,
  ): AssistantMessageEventStream =>
    streamCocore(model, context, options, {
      routing,
      country: routingPrefs.country,
      minProviderVersion: routingPrefs.minProviderVersion,
    });
}

async function registerCocoreProvider(
  pi: ExtensionAPI,
  apiKey: string,
  routingPrefs: { country?: string; minProviderVersion?: string } = {},
): Promise<number> {
  const response = await fetch(`${BASE_URL}/models`, {
    headers: { Authorization: `Bearer ${apiKey}` },
  });

  if (!response.ok) {
    const body = await response.text().catch(() => "");
    throw new Error(
      `Co/Core /models returned ${response.status} ${response.statusText}: ${body.slice(0, 200)}`
    );
  }

  const payload = (await response.json()) as CocoreModelsResponse;

  if (payload.object !== "list" || !Array.isArray(payload.data)) {
    throw new Error("[cocore] Unexpected /models response format");
  }

  const models = payload.data
    .filter((m) => m.id !== "stub") // Skip internal stub entry
    .map((m) => {
      const caps = deriveModelCapabilities(m.id);
      return {
        id: m.id,
        name: m.id.split("/").pop() ?? m.id,
        ...caps,
      };
    });

  // Register one provider per routing tier. Same model list, different
  // URL paths. The verified tier implicitly sends `min_trust:
  // hardware-attested` via buildCocoreRequestBody so unconfigured users
  // land on cryptographically-verified providers by default.
  for (const routing of ROUTINGS) {
    pi.registerProvider(providerKey(routing), {
      name: providerName(routing),
      baseUrl: BASE_URL,
      apiKey,
      api: "openai-completions",
      models,
      streamSimple: makeStreamForRouting(routing, routingPrefs),
    });
  }

  return models.length;
}

// ── Get model family for a cocore model ──────────────────────────────────────

// Exact parser pairings audited in https://github.com/graze-social/cocore/pull/196.
// Other backends and quantizations can fail the canary even within the same family.
const VERIFIED_TOOL_MODEL_IDS = new Set([
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
  "leonsarmiento/Ornith-1.0-35B-5bit-mlx",
  "mlx-community/Qwen2.5-0.5B-Instruct-4bit",
  "mlx-community/Qwen2.5-3B-Instruct-4bit",
  "mlx-community/Qwen2.5-7B-Instruct-4bit",
  "mlx-community/Qwen2.5-32B-Instruct-4bit",
].map((id) => id.toLowerCase()));

function isVerifiedToolModel(modelId: string): boolean {
  return VERIFIED_TOOL_MODEL_IDS.has(modelId.toLowerCase());
}

/**
 * Map a cocore model id to its tool-calling family. Returns null for
 * verified-tool models — those bypass the text-envelope path because
 * the server returns structured `tool_calls` directly.
 */
function getModelFamily(modelId: string): "gemma" | "qwen" | null {
  if (isVerifiedToolModel(modelId)) return null;
  const id = modelId.toLowerCase();
  if (id.includes("gemma")) return "gemma";
  if (id.includes("qwen")) return "qwen";
  return null;
}

// ── Extension entry point ────────────────────────────────────────────────────

export default async function (pi: ExtensionAPI) {
  const config = loadConfig();

  // ── Happy path: API key already saved ──────────────────────────────────
  if (config?.apiKey) {
    try {
      const count = await registerCocoreProvider(pi, config.apiKey, {
        country: config.country,
        minProviderVersion: config.minProviderVersion,
      });
      console.log(`[cocore] Registered ${count} model(s) across ${ROUTINGS.length} routing tiers`);
    } catch (err) {
      console.error(`[cocore] ${err instanceof Error ? err.message : err}`);
    }
    registerEventHandlers(pi);
    return;
  }

  // ── First run: prompt for API key on session start ─────────────────────
  let setupDone = false;

  pi.on("session_start", async (_event, ctx) => {
    if (setupDone) return;
    setupDone = true;

    // Re-check config in case another process saved it
    const fresh = loadConfig();
    if (fresh?.apiKey) {
      try {
        await registerCocoreProvider(pi, fresh.apiKey, {
          country: fresh.country,
          minProviderVersion: fresh.minProviderVersion,
        });
        ctx.ui.notify("Co/Core provider registered!", "info");
      } catch (err) {
        ctx.ui.notify(
          `Co/Core registration failed: ${err instanceof Error ? err.message : err}`,
          "error"
        );
      }
      registerEventHandlers(pi);
      return;
    }

    const apiKey = await ctx.ui.input(
      "Enter your Co/Core API key (from cocore.dev):",
      { password: true }
    );

    if (!apiKey?.trim()) {
      ctx.ui.notify(
        "Co/Core setup skipped. Run /cocore-setup when ready.",
        "warning"
      );
      return;
    }

    saveConfig({ apiKey: apiKey.trim() });

    try {
      const count = await registerCocoreProvider(pi, apiKey.trim());
      ctx.ui.notify(
        `Co/Core ready — ${count} model(s) available across ${ROUTINGS.length} routing tiers.`,
        "info",
      );
      console.log(`[cocore] Registered ${count} model(s) across ${ROUTINGS.length} routing tiers`);
    } catch (err) {
      ctx.ui.notify(
        `Co/Core: ${err instanceof Error ? err.message : err}`,
        "error"
      );
    }

    registerEventHandlers(pi);
  });

  // Register event handlers immediately (setup command still works)
  registerEventHandlers(pi);
}

/**
 * Register all event handlers. Called both on happy path and after setup.
 */
let handlersRegistered = false;

function registerEventHandlers(pi: ExtensionAPI) {
  if (handlersRegistered) return;
  handlersRegistered = true;

  // ── Tool-calling instructions are now injected in the custom stream ───
  // The streamCocore function handles Gemma/Qwen tool instruction injection
  // and strips the tools array before sending to the API.

  // ── Fix tool calls in text content for all cocore models ──────────────
  // Gemma and Qwen models output tool calls as text tokens instead of
  // structured tool_calls. We detect and convert them after streaming.
  pi.on("message_end", (event) => {
    const modelId = (event.message as any).model as string | undefined;
    const provider = (event.message as any).provider as string | undefined;

    if (!isCocoreModel({ provider })) return;

    const family = getModelFamily(modelId ?? "");
    if (!family) return; // Not a Gemma or Qwen model

    const fixed = fixCocoreToolCalls(event.message as any, family);
    if (fixed === event.message) return;

    // Return the modified message to replace the original
    return { message: fixed as any };
  });

  // ── Manual setup command ───────────────────────────────────────────────
  pi.registerCommand("cocore-setup", {
    description: "Configure or reconfigure your Co/Core API key, country, and minimum provider version",
    handler: async (_args, ctx) => {
      const existing = loadConfig();
      const prompt = existing?.apiKey
        ? "Enter a new Co/Core API key (leave blank to keep current):"
        : "Enter your Co/Core API key (from cocore.dev):";

      const apiKey = await ctx.ui.input(prompt, { password: true });
      const trimmedKey = apiKey?.trim();

      const countryPrompt = existing?.country
        ? `Country code (ISO 3166-1 alpha-2, blank to keep "${existing.country}", - to clear):`
        : "Country code (ISO 3166-1 alpha-2, e.g. US; blank for none):";
      const countryInput = await ctx.ui.input(countryPrompt);
      const country = resolveOptionalConfigInput(countryInput, existing?.country);

      const versionPrompt = existing?.minProviderVersion
        ? `Minimum tray-provider release (blank to keep "${existing.minProviderVersion}", - to clear):`
        : "Minimum tray-provider release, e.g. 0.9.32 (blank for none):";
      const versionInput = await ctx.ui.input(versionPrompt);
      const minProviderVersion = resolveOptionalConfigInput(
        versionInput,
        existing?.minProviderVersion,
      );

      const nextKey = trimmedKey || existing?.apiKey;
      if (!nextKey) {
        ctx.ui.notify("Co/Core setup skipped.", "warning");
        return;
      }

      const next: CocoreConfig = { apiKey: nextKey };
      if (country) next.country = country;
      if (minProviderVersion) next.minProviderVersion = minProviderVersion;
      saveConfig(next);
      const saved = trimmedKey ? "API key" : "settings";
      ctx.ui.notify(
        `Co/Core ${saved} saved. Restart pi or run /reload to activate.`,
        "info",
      );
    },
  });
}

// Internal exports for unit tests. The default extension export above is
// what pi loads; these named exports let tests exercise the
// parsing/conversion logic without spinning up the full extension.
export {
  streamCocore,
  convertMessagesForOpenAI,
  buildCocoreRequestBody,
  getModelFamily,
  isVerifiedToolModel,
  parseToolCalls,
  parseGemma4ToolCalls,
  parseGemma3ToolCalls,
  parsePositionalGemmaCalls,
  fixCocoreToolCalls,
  normalizeModelText,
  stripThinkingContent,
  parseGemmaPseudoJson,
  parseServerError,
  extractErrorCode,
  friendlyMessageFor,
  chatCompletionsPath,
  providerKey,
  providerName,
  ROUTINGS,
};
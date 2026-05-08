import * as Effect from "effect/Effect";
import * as Layer from "effect/Layer";
import * as Stream from "effect/Stream";
import {
  AiError,
  IdGenerator,
  LanguageModel as AiLanguageModel,
  Prompt,
  Response,
  Tool,
} from "effect/unstable/ai";
import * as Sse from "effect/unstable/encoding/Sse";
import { WorkerEnvironment } from "../Workers/Worker.ts";
import type { AiGatewayClient } from "./AiGatewayBinding.ts";

/**
 * Options for constructing an AI Gateway-backed Workers AI LanguageModel.
 */
export interface Options {
  /** Already-bound AI Gateway client from `AiGatewayBinding.bind(gateway)`. */
  readonly client: AiGatewayClient;
  /** Workers AI model id, e.g. `@cf/meta/llama-3.3-70b-instruct-fp8-fast`. */
  readonly model: string;
  /** Optional per-call defaults; overridable per request via `providerOptions`. */
  readonly parameters?: {
    readonly temperature?: number;
    readonly maxTokens?: number;
    readonly topP?: number;
    readonly topK?: number;
    readonly seed?: number;
    readonly frequencyPenalty?: number;
    readonly presencePenalty?: number;
  };
}

// ---------------------------------------------------------------------------
// Wire format types (Workers AI request/response)
//
// Workers AI mixes two response shapes:
//   - Native:  { response: "...", tool_calls: [...] , usage }
//   - OpenAI:  { choices: [{ message: { content, tool_calls, reasoning_content } }], usage }
//
// We accept both defensively — schemas would over-constrain.
// ---------------------------------------------------------------------------

interface WorkersAiMessage {
  readonly role: "system" | "user" | "assistant" | "tool";
  readonly content?: unknown;
  readonly name?: string;
  readonly tool_call_id?: string;
  readonly tool_calls?: ReadonlyArray<{
    readonly id: string;
    readonly type: "function";
    readonly function: { readonly name: string; readonly arguments: string };
  }>;
  readonly reasoning?: string;
}

interface WorkersAiToolDef {
  readonly type: "function";
  readonly function: {
    readonly name: string;
    readonly description?: string;
    readonly parameters: unknown;
  };
}

type WorkersAiToolChoice =
  | "auto"
  | "required"
  | "none"
  | { readonly type: "function"; readonly function: { readonly name: string } };

interface WorkersAiInputs {
  readonly messages: ReadonlyArray<WorkersAiMessage>;
  readonly tools?: ReadonlyArray<WorkersAiToolDef>;
  readonly tool_choice?: WorkersAiToolChoice;
  readonly stream?: boolean;
  readonly max_tokens?: number;
  readonly temperature?: number;
  readonly top_p?: number;
  readonly top_k?: number;
  readonly random_seed?: number;
  readonly frequency_penalty?: number;
  readonly presence_penalty?: number;
}

// ---------------------------------------------------------------------------
// Prompt → Workers AI messages
// ---------------------------------------------------------------------------

const uint8ArrayToBase64 = (bytes: Uint8Array): string => {
  let binary = "";
  const chunkSize = 8192;
  for (let i = 0; i < bytes.length; i += chunkSize) {
    const chunk = bytes.subarray(i, Math.min(i + chunkSize, bytes.length));
    binary += String.fromCharCode(...chunk);
  }
  return btoa(binary);
};

const fileToImageUrl = (
  data: string | Uint8Array | URL,
  mediaType: string,
): string => {
  if (data instanceof URL) return data.toString();
  if (data instanceof Uint8Array) {
    return `data:${mediaType};base64,${uint8ArrayToBase64(data)}`;
  }
  if (data.startsWith("data:") || data.startsWith("http")) return data;
  return `data:${mediaType};base64,${data}`;
};

const convertPromptToMessages = (
  prompt: Prompt.Prompt,
): ReadonlyArray<WorkersAiMessage> => {
  const messages: Array<WorkersAiMessage> = [];

  for (const message of prompt.content) {
    switch (message.role) {
      case "system": {
        messages.push({ role: "system", content: message.content });
        break;
      }
      case "user": {
        const textParts: Array<string> = [];
        const imageParts: Array<{
          type: "image_url";
          image_url: { url: string };
        }> = [];
        for (const part of message.content) {
          if (part.type === "text") {
            textParts.push(part.text);
          } else if (part.type === "file") {
            imageParts.push({
              type: "image_url",
              image_url: {
                url: fileToImageUrl(part.data, part.mediaType),
              },
            });
          }
        }
        if (imageParts.length > 0) {
          const content: Array<unknown> = [];
          if (textParts.length > 0) {
            content.push({ type: "text", text: textParts.join("\n") });
          }
          content.push(...imageParts);
          messages.push({ role: "user", content });
        } else {
          messages.push({ role: "user", content: textParts.join("\n") });
        }
        break;
      }
      case "assistant": {
        let text = "";
        let reasoning = "";
        const toolCalls: Array<{
          id: string;
          type: "function";
          function: { name: string; arguments: string };
        }> = [];
        for (const part of message.content) {
          switch (part.type) {
            case "text":
              text += part.text;
              break;
            case "reasoning":
              reasoning += part.text;
              break;
            case "tool-call":
              toolCalls.push({
                id: part.id,
                type: "function",
                function: {
                  name: part.name,
                  arguments:
                    typeof part.params === "string"
                      ? part.params
                      : JSON.stringify(part.params ?? {}),
                },
              });
              break;
            default:
              break;
          }
        }
        messages.push({
          role: "assistant",
          content: text,
          ...(reasoning ? { reasoning } : {}),
          ...(toolCalls.length > 0 ? { tool_calls: toolCalls } : {}),
        });
        break;
      }
      case "tool": {
        for (const part of message.content) {
          if (part.type === "tool-result") {
            const result = part.result;
            const content =
              typeof result === "string" ? result : JSON.stringify(result);
            messages.push({
              role: "tool",
              name: part.name,
              tool_call_id: part.id,
              content,
            });
          }
        }
        break;
      }
    }
  }
  return messages;
};

// ---------------------------------------------------------------------------
// Tools / tool_choice
// ---------------------------------------------------------------------------

const prepareTools = (
  tools: ReadonlyArray<Tool.Any>,
  toolChoice: AiLanguageModel.ProviderOptions["toolChoice"],
): {
  tools?: ReadonlyArray<WorkersAiToolDef>;
  tool_choice?: WorkersAiToolChoice;
} => {
  if (tools.length === 0) {
    return {};
  }
  const mapped: Array<WorkersAiToolDef> = tools.map((tool) => ({
    type: "function",
    function: {
      name: tool.name,
      description: Tool.getDescription(tool),
      parameters: Tool.getJsonSchema(tool),
    },
  }));

  if (toolChoice === "auto" || toolChoice == null) {
    return { tools: mapped, tool_choice: "auto" };
  }
  if (toolChoice === "none") {
    return { tools: mapped, tool_choice: "none" };
  }
  if (toolChoice === "required") {
    return { tools: mapped, tool_choice: "required" };
  }
  if (typeof toolChoice === "object" && "tool" in toolChoice) {
    return {
      tools: mapped.filter((t) => t.function.name === toolChoice.tool),
      tool_choice: "required",
    };
  }
  if (typeof toolChoice === "object" && "oneOf" in toolChoice) {
    const allowed = new Set(toolChoice.oneOf);
    return {
      tools: mapped.filter((t) => allowed.has(t.function.name)),
      tool_choice: toolChoice.mode === "required" ? "required" : "auto",
    };
  }
  return { tools: mapped, tool_choice: "auto" };
};

// ---------------------------------------------------------------------------
// Request body
// ---------------------------------------------------------------------------

const toRequestBody = ({
  options,
  parameters,
  stream,
}: {
  readonly options: AiLanguageModel.ProviderOptions;
  readonly parameters: Options["parameters"];
  readonly stream: boolean;
}): WorkersAiInputs => {
  const messages = convertPromptToMessages(options.prompt);
  const { tools, tool_choice } = prepareTools(
    options.tools,
    options.toolChoice,
  );
  return {
    messages,
    ...(tools !== undefined ? { tools } : {}),
    ...(tool_choice !== undefined ? { tool_choice } : {}),
    ...(stream ? { stream: true } : {}),
    ...(parameters?.maxTokens !== undefined
      ? { max_tokens: parameters.maxTokens }
      : {}),
    ...(parameters?.temperature !== undefined
      ? { temperature: parameters.temperature }
      : {}),
    ...(parameters?.topP !== undefined ? { top_p: parameters.topP } : {}),
    ...(parameters?.topK !== undefined ? { top_k: parameters.topK } : {}),
    ...(parameters?.seed !== undefined ? { random_seed: parameters.seed } : {}),
    ...(parameters?.frequencyPenalty !== undefined
      ? { frequency_penalty: parameters.frequencyPenalty }
      : {}),
    ...(parameters?.presencePenalty !== undefined
      ? { presence_penalty: parameters.presencePenalty }
      : {}),
  };
};

// ---------------------------------------------------------------------------
// Finish reason / usage mapping
// ---------------------------------------------------------------------------

const mapFinishReason = (raw: unknown): Response.FinishReason => {
  switch (raw) {
    case "stop":
      return "stop";
    case "length":
    case "model_length":
      return "length";
    case "tool_calls":
      return "tool-calls";
    case "content_filter":
    case "content-filter":
      return "content-filter";
    case "error":
      return "error";
    case undefined:
    case null:
      return "unknown";
    default:
      return "other";
  }
};

const mapUsage = (
  raw: Record<string, unknown> | undefined,
): typeof Response.Usage.Encoded => {
  const usage = (raw?.usage as Record<string, unknown> | undefined) ?? {};
  const promptTokens = (usage.prompt_tokens as number | undefined) ?? 0;
  const completionTokens = (usage.completion_tokens as number | undefined) ?? 0;
  const cached = (
    usage.prompt_tokens_details as { cached_tokens?: number } | undefined
  )?.cached_tokens;
  return {
    inputTokens: {
      uncached:
        cached !== undefined ? Math.max(0, promptTokens - cached) : undefined,
      total: promptTokens,
      cacheRead: cached,
      cacheWrite: undefined,
    },
    outputTokens: {
      total: completionTokens,
      text: undefined,
      reasoning: undefined,
    },
  };
};

// ---------------------------------------------------------------------------
// generateText: JSON response → Response.PartEncoded[]
// ---------------------------------------------------------------------------

const extractText = (output: Record<string, unknown>): string | undefined => {
  const choices = output.choices as
    | Array<{ message?: { content?: string | null } }>
    | undefined;
  const choiceContent = choices?.[0]?.message?.content;
  if (choiceContent != null && String(choiceContent).length > 0) {
    return String(choiceContent);
  }
  if ("response" in output) {
    const r = output.response;
    if (r == null) return undefined;
    if (typeof r === "object") return JSON.stringify(r);
    return String(r);
  }
  return undefined;
};

const extractReasoning = (
  output: Record<string, unknown>,
): string | undefined => {
  const choices = output.choices as
    | Array<{ message?: { reasoning_content?: string; reasoning?: string } }>
    | undefined;
  const r =
    choices?.[0]?.message?.reasoning_content ??
    choices?.[0]?.message?.reasoning;
  return r && r.length > 0 ? r : undefined;
};

const extractToolCalls = (
  output: Record<string, unknown>,
): Array<{ id: string; name: string; arguments: unknown }> => {
  const collect = (
    raw: ReadonlyArray<Record<string, unknown>>,
  ): Array<{ id: string; name: string; arguments: unknown }> =>
    raw.flatMap((tc) => {
      const fn = tc.function as
        | { name?: string; arguments?: unknown }
        | undefined;
      const id = (tc.id as string | undefined) ?? "";
      if (fn?.name) {
        return [{ id, name: fn.name, arguments: fn.arguments ?? "" }];
      }
      const flatName = tc.name as string | undefined;
      if (flatName) {
        return [{ id, name: flatName, arguments: tc.arguments ?? "" }];
      }
      return [];
    });

  if (Array.isArray(output.tool_calls)) {
    return collect(output.tool_calls as ReadonlyArray<Record<string, unknown>>);
  }
  const choices = output.choices as
    | Array<{
        message?: { tool_calls?: ReadonlyArray<Record<string, unknown>> };
      }>
    | undefined;
  const fromChoice = choices?.[0]?.message?.tool_calls;
  if (Array.isArray(fromChoice)) {
    return collect(fromChoice);
  }
  return [];
};

const parseGenerateText = Effect.fnUntraced(function* (
  raw: Record<string, unknown>,
) {
  const idGenerator = yield* IdGenerator.IdGenerator;
  const parts: Array<Response.PartEncoded> = [];

  const reasoning = extractReasoning(raw);
  if (reasoning !== undefined) {
    parts.push({
      type: "reasoning",
      text: reasoning,
    });
  }

  const text = extractText(raw);
  if (text !== undefined && text.length > 0) {
    parts.push({ type: "text", text });
  }

  const toolCalls = extractToolCalls(raw);
  for (const tc of toolCalls) {
    const id = tc.id || (yield* idGenerator.generateId());
    let params: unknown = tc.arguments;
    if (typeof params === "string") {
      try {
        params = JSON.parse(params);
      } catch {
        // leave as raw string; framework's tool-result decoder will fail loudly
      }
    }
    parts.push({
      type: "tool-call",
      id,
      name: tc.name,
      params,
    });
  }

  const finishReason = mapFinishReason(
    extractFinishReason(raw) ?? (toolCalls.length > 0 ? "tool_calls" : "stop"),
  );

  parts.push({
    type: "finish",
    reason: finishReason,
    usage: mapUsage(raw),
    response: undefined,
  });

  return parts;
});

const extractFinishReason = (
  output: Record<string, unknown>,
): string | undefined => {
  const choices = output.choices as
    | Array<{ finish_reason?: string }>
    | undefined;
  return (
    choices?.[0]?.finish_reason ?? (output.finish_reason as string | undefined)
  );
};

// ---------------------------------------------------------------------------
// streamText: SSE byte stream → Stream<Response.StreamPartEncoded>
// ---------------------------------------------------------------------------

interface StreamState {
  readonly textId: string | undefined;
  readonly reasoningId: string | undefined;
  readonly toolCalls: ReadonlyMap<
    number,
    { readonly id: string; readonly name: string }
  >;
  readonly lastToolIndex: number | undefined;
  readonly closedToolIndices: ReadonlySet<number>;
  readonly usage: Record<string, unknown> | undefined;
  readonly finishReason: string | undefined;
  readonly receivedAnyData: boolean;
  readonly receivedDone: boolean;
}

const initialStreamState = (): StreamState => ({
  textId: undefined,
  reasoningId: undefined,
  toolCalls: new Map(),
  lastToolIndex: undefined,
  closedToolIndices: new Set(),
  usage: undefined,
  finishReason: undefined,
  receivedAnyData: false,
  receivedDone: false,
});

const isNullFinalizationToolCall = (tc: Record<string, unknown>): boolean => {
  const fn = tc.function as Record<string, unknown> | undefined;
  const name = fn?.name ?? tc.name ?? null;
  const args = fn?.arguments ?? tc.arguments ?? null;
  const id = tc.id ?? null;
  return !id && !name && (!args || args === "");
};

const closeToolCall = (
  state: StreamState,
  index: number,
  out: Array<Response.StreamPartEncoded>,
): StreamState => {
  if (state.closedToolIndices.has(index)) return state;
  const tc = state.toolCalls.get(index);
  if (!tc) return state;
  out.push({ type: "tool-params-end", id: tc.id });
  // Mark as closed; framework re-assembles `tool-call` from accumulated deltas.
  const closed = new Set(state.closedToolIndices);
  closed.add(index);
  return { ...state, closedToolIndices: closed };
};

const handleToolCallChunks = (
  state: StreamState,
  chunks: ReadonlyArray<Record<string, unknown>>,
  idGenerator: IdGenerator.Service,
  out: Array<Response.StreamPartEncoded>,
): Effect.Effect<StreamState> =>
  Effect.gen(function* () {
    let cur = state;
    for (const tc of chunks) {
      if (isNullFinalizationToolCall(tc)) {
        if (cur.lastToolIndex !== undefined) {
          cur = closeToolCall(cur, cur.lastToolIndex, out);
        }
        continue;
      }
      const tcIndex = (tc.index as number | undefined) ?? 0;
      const fn = tc.function as
        | { name?: string; arguments?: string }
        | undefined;
      const tcName = fn?.name ?? (tc.name as string | undefined) ?? "";
      const tcArgs =
        fn?.arguments ?? (tc.arguments as string | undefined) ?? "";
      const tcIdRaw = (tc.id as string | undefined) ?? "";

      let entry = cur.toolCalls.get(tcIndex);
      if (entry === undefined) {
        // Close the previous active call before starting a new one.
        if (cur.lastToolIndex !== undefined && cur.lastToolIndex !== tcIndex) {
          cur = closeToolCall(cur, cur.lastToolIndex, out);
        }
        const id = tcIdRaw || (yield* idGenerator.generateId());
        entry = { id, name: tcName };
        const next = new Map(cur.toolCalls);
        next.set(tcIndex, entry);
        cur = { ...cur, toolCalls: next, lastToolIndex: tcIndex };
        out.push({
          type: "tool-params-start",
          id: entry.id,
          name: entry.name,
        });
        if (tcArgs.length > 0) {
          out.push({
            type: "tool-params-delta",
            id: entry.id,
            delta: tcArgs,
          });
        }
      } else {
        cur = { ...cur, lastToolIndex: tcIndex };
        if (tcArgs.length > 0) {
          out.push({
            type: "tool-params-delta",
            id: entry.id,
            delta: tcArgs,
          });
        }
      }
    }
    return cur;
  });

const handleStreamChunk = (
  state: StreamState,
  data: string,
  idGenerator: IdGenerator.Service,
): Effect.Effect<
  readonly [StreamState, ReadonlyArray<Response.StreamPartEncoded>]
> =>
  Effect.gen(function* () {
    const out: Array<Response.StreamPartEncoded> = [];

    if (data === "" || data === "[DONE]") {
      return [
        data === "[DONE]" ? { ...state, receivedDone: true } : state,
        out,
      ] as const;
    }

    let chunk: Record<string, unknown>;
    try {
      chunk = JSON.parse(data) as Record<string, unknown>;
    } catch {
      return [state, out] as const;
    }

    let cur: StreamState = { ...state, receivedAnyData: true };
    if (chunk.usage !== undefined) {
      cur = { ...cur, usage: chunk as Record<string, unknown> };
    }

    const choices = chunk.choices as
      | Array<{ finish_reason?: string; delta?: Record<string, unknown> }>
      | undefined;
    const choiceFinish = choices?.[0]?.finish_reason;
    const directFinish = chunk.finish_reason as string | undefined;
    if (choiceFinish != null) cur = { ...cur, finishReason: choiceFinish };
    else if (directFinish != null) cur = { ...cur, finishReason: directFinish };

    // Native: top-level `response`
    const nativeResponse = chunk.response;
    if (
      nativeResponse !== undefined &&
      nativeResponse !== null &&
      nativeResponse !== ""
    ) {
      const text = String(nativeResponse);
      if (text.length > 0) {
        if (cur.reasoningId !== undefined) {
          out.push({ type: "reasoning-end", id: cur.reasoningId });
          cur = { ...cur, reasoningId: undefined };
        }
        if (cur.textId === undefined) {
          const id = yield* idGenerator.generateId();
          cur = { ...cur, textId: id };
          out.push({ type: "text-start", id });
        }
        out.push({ type: "text-delta", id: cur.textId!, delta: text });
      }
    }

    // Native: top-level `tool_calls`
    if (Array.isArray(chunk.tool_calls)) {
      if (cur.reasoningId !== undefined) {
        out.push({ type: "reasoning-end", id: cur.reasoningId });
        cur = { ...cur, reasoningId: undefined };
      }
      cur = yield* handleToolCallChunks(
        cur,
        chunk.tool_calls as ReadonlyArray<Record<string, unknown>>,
        idGenerator,
        out,
      );
    }

    // OpenAI: choices[0].delta
    const delta = choices?.[0]?.delta;
    if (delta) {
      const reasoningDelta = (delta.reasoning_content ?? delta.reasoning) as
        | string
        | undefined;
      if (reasoningDelta && reasoningDelta.length > 0) {
        if (cur.reasoningId === undefined) {
          const id = yield* idGenerator.generateId();
          cur = { ...cur, reasoningId: id };
          out.push({ type: "reasoning-start", id });
        }
        out.push({
          type: "reasoning-delta",
          id: cur.reasoningId!,
          delta: reasoningDelta,
        });
      }

      const textDelta = delta.content as string | undefined;
      if (textDelta && textDelta.length > 0) {
        if (cur.reasoningId !== undefined) {
          out.push({ type: "reasoning-end", id: cur.reasoningId });
          cur = { ...cur, reasoningId: undefined };
        }
        if (cur.textId === undefined) {
          const id = yield* idGenerator.generateId();
          cur = { ...cur, textId: id };
          out.push({ type: "text-start", id });
        }
        out.push({ type: "text-delta", id: cur.textId!, delta: textDelta });
      }

      const deltaToolCalls = delta.tool_calls as
        | ReadonlyArray<Record<string, unknown>>
        | undefined;
      if (Array.isArray(deltaToolCalls)) {
        if (cur.reasoningId !== undefined) {
          out.push({ type: "reasoning-end", id: cur.reasoningId });
          cur = { ...cur, reasoningId: undefined };
        }
        cur = yield* handleToolCallChunks(
          cur,
          deltaToolCalls,
          idGenerator,
          out,
        );
      }
    }

    return [cur, out] as const;
  });

const finalizeStream = (
  state: StreamState,
): ReadonlyArray<Response.StreamPartEncoded> => {
  const out: Array<Response.StreamPartEncoded> = [];
  let cur = state;
  for (const [idx] of cur.toolCalls) {
    cur = closeToolCall(cur, idx, out);
  }
  if (cur.reasoningId !== undefined) {
    out.push({ type: "reasoning-end", id: cur.reasoningId });
  }
  if (cur.textId !== undefined) {
    out.push({ type: "text-end", id: cur.textId });
  }

  const reason: Response.FinishReason =
    !cur.receivedDone && cur.receivedAnyData && cur.finishReason === undefined
      ? "error"
      : mapFinishReason(cur.finishReason);

  out.push({
    type: "finish",
    reason,
    usage: mapUsage(cur.usage),
    response: undefined,
  });
  return out;
};

const parseStreamText = (
  resp: Response,
  idGenerator: IdGenerator.Service,
): Stream.Stream<Response.StreamPartEncoded, AiError.AiError> => {
  const body = resp.body;
  if (body === null) {
    return Stream.fromIterable<Response.StreamPartEncoded>(
      finalizeStream(initialStreamState()),
    );
  }
  return Stream.fromReadableStream<Uint8Array, AiError.AiError>({
    evaluate: () => body,
    onError: (cause) => toAiError(cause, "streamText"),
  }).pipe(
    Stream.decodeText(),
    Stream.pipeThroughChannel(Sse.decode<AiError.AiError, unknown>()),
    Stream.catchTag("Retry", (retry) => Stream.die(retry)),
    Stream.mapAccumEffect(
      initialStreamState,
      (state, event) => handleStreamChunk(state, event.data, idGenerator),
      { onHalt: (state) => finalizeStream(state) },
    ),
  );
};

// ---------------------------------------------------------------------------
// make / layer
// ---------------------------------------------------------------------------

const toAiError = (
  cause: unknown,
  method: "generateText" | "streamText",
): AiError.AiError =>
  AiError.AiError.make({
    module: "Cloudflare.AiGateway.LanguageModel",
    method,
    reason: new AiError.UnknownError({
      description:
        cause instanceof Error ? cause.message : "AI Gateway request failed",
    }),
  });

/**
 * Build a {@link AiLanguageModel.Service} that proxies generateText/streamText
 * through the supplied AI Gateway client to a Workers AI model.
 */
export const make = ({
  client,
  model,
  parameters,
}: Options): Effect.Effect<AiLanguageModel.Service, never, WorkerEnvironment> =>
  Effect.gen(function* () {
    const env = yield* WorkerEnvironment;
    const ai = yield* client.raw;
    const gatewayId = yield* client.id;

    const callRaw = (
      body: WorkersAiInputs,
      method: "generateText" | "streamText",
    ): Effect.Effect<Response, AiError.AiError> =>
      Effect.tryPromise({
        try: () =>
          ai.run(
            model as keyof AiModels,
            body as unknown as AiModels[keyof AiModels]["inputs"],
            {
              gateway: { id: gatewayId },
              returnRawResponse: true,
            },
          ),
        catch: (cause) => toAiError(cause, method),
      }).pipe(Effect.provideService(WorkerEnvironment, env));

    return yield* AiLanguageModel.make({
      generateText: (options) =>
        Effect.gen(function* () {
          const body = toRequestBody({ options, parameters, stream: false });
          const resp = yield* callRaw(body, "generateText");
          const json = yield* Effect.tryPromise({
            try: () => resp.json() as Promise<Record<string, unknown>>,
            catch: (cause) => toAiError(cause, "generateText"),
          });
          return yield* parseGenerateText(json);
        }),
      streamText: (options) =>
        Stream.unwrap(
          Effect.gen(function* () {
            const idGenerator = yield* IdGenerator.IdGenerator;
            const body = toRequestBody({ options, parameters, stream: true });
            const resp = yield* callRaw(body, "streamText");
            return parseStreamText(resp, idGenerator);
          }),
        ),
    });
  });

/**
 * Provide a {@link AiLanguageModel.LanguageModel} layer backed by the supplied
 * AI Gateway client and Workers AI model.
 */
export const layer = (
  options: Options,
): Layer.Layer<AiLanguageModel.LanguageModel, never, WorkerEnvironment> =>
  Layer.effect(AiLanguageModel.LanguageModel, make(options));

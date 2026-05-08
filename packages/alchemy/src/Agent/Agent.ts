import * as Cloudflare from "alchemy/Cloudflare";
import * as Effect from "effect/Effect";
import type * as Schema from "effect/Schema";
import * as S from "effect/Schema";
import * as Stream from "effect/Stream";
import type { AiError } from "effect/unstable/ai";
import {
  Tool as AiTool,
  LanguageModel,
  Prompt,
  Response,
  Toolkit,
} from "effect/unstable/ai";
import { HttpServerResponse } from "effect/unstable/http";

// ---------------------------------------------------------------------------
// Public type surface (do not change without asking)
// ---------------------------------------------------------------------------

export const Tool: {
  <Self>(): <Id extends string>(
    id: Id,
  ) => <References extends any[]>(
    template: TemplateStringsArray,
    ...references: References
  ) => any;
  <Id extends string>(
    id: Id,
  ): <References extends any[]>(
    template: TemplateStringsArray,
    ...references: References
  ) => any;
} = ((...args: any[]): any => {
  // Tool<Self>() — returns the curried tag form
  if (args.length === 0) {
    return toolTagForm;
  }
  // Tool(id) — returns the inline form (template) => (handler) => Class
  return toolInlineForm(args[0] as string);
}) as any;

export interface AgentInstance {
  /**
   * Generate text using the language model, with the agent's system prompt
   * and tools baked in. Mirrors `LanguageModel.generateText` (no-toolkit form).
   */
  generateText(options: {
    readonly prompt: Prompt.RawInput;
    readonly toolChoice?: LanguageModel.ToolChoice<string> | undefined;
    readonly concurrency?: number | "unbounded" | "inherit" | undefined;
    readonly disableToolCallResolution?: boolean | undefined;
  }): Effect.Effect<
    LanguageModel.GenerateTextResponse<any>,
    AiError.AiError,
    LanguageModel.LanguageModel
  >;

  /**
   * Generate text using the language model with streaming output, with the
   * agent's system prompt and tools baked in. Mirrors `LanguageModel.streamText`
   * (no-toolkit form).
   */
  streamText(options: {
    readonly prompt: Prompt.RawInput;
    readonly toolChoice?: LanguageModel.ToolChoice<string> | undefined;
    readonly concurrency?: number | "unbounded" | "inherit" | undefined;
    readonly disableToolCallResolution?: boolean | undefined;
  }): Stream.Stream<
    Response.StreamPart<any>,
    AiError.AiError,
    LanguageModel.LanguageModel
  >;
}

export const Agent: {
  <Self>(): <Id extends string>(
    id: Id,
  ) => <References extends any[]>(
    template: TemplateStringsArray,
    ...references: References
  ) => Effect.Effect<Self> & {
    new (_: never): AgentInstance;
  };
  <Id extends string>(
    id: Id,
  ): <References extends any[]>(
    template: TemplateStringsArray,
    ...references: References
  ) => Effect.Effect<any> & {
    new (_: never): AgentInstance;
  };
} = ((...args: any[]): any => {
  if (args.length === 0) return agentTagForm;
  return agentInlineForm(args[0] as string);
}) as any;

// ---------------------------------------------------------------------------
// Runtime — Tool
// ---------------------------------------------------------------------------

const AgentToolBrand = Symbol.for("alchemy/Agent/Tool");

interface BrandedTool {
  readonly [AgentToolBrand]: true;
  readonly tool: AiTool.Any;
  readonly handler: (params: any, ctx: any) => Effect.Effect<any, any, any>;
}

const isBrandedTool = (x: unknown): x is BrandedTool =>
  typeof x === "object" &&
  x !== null &&
  ((x as any)[AgentToolBrand] === true ||
    (x as any)?.constructor?.[AgentToolBrand] === true);

const buildToolDef = (
  id: string,
  template: TemplateStringsArray,
  refs: ReadonlyArray<unknown>,
) => {
  const description = renderTemplate(template, refs);
  const parameters = mergeStructs(refs.filter(isSchemaStruct));
  return AiTool.make(id, {
    description,
    ...(parameters !== undefined ? { parameters } : {}),
  });
};

const wrapHandler = (handler: any) => (params: any, ctx: any) =>
  Effect.flatMap(handler, (run: any) => run(params, ctx));

const toolInlineForm =
  (id: string) =>
  (template: TemplateStringsArray, ...refs: ReadonlyArray<unknown>) =>
  (handlerEffect: any) => {
    const tool = buildToolDef(id, template, refs);
    const handler = wrapHandler(handlerEffect);
    class ToolClass {
      static readonly [AgentToolBrand] = true;
      static readonly tool = tool;
      static readonly handler = handler;
    }
    return ToolClass;
  };

const toolTagForm =
  () =>
  (id: string) =>
  (template: TemplateStringsArray, ...refs: ReadonlyArray<unknown>) => {
    const tool = buildToolDef(id, template, refs);
    class ToolTag {
      static readonly [AgentToolBrand] = true;
      static readonly tool = tool;
      static handler: BrandedTool["handler"] | undefined;
      static make(handlerEffect: any) {
        ToolTag.handler = wrapHandler(handlerEffect);
        return ToolTag;
      }
    }
    return ToolTag;
  };

// ---------------------------------------------------------------------------
// Runtime — Agent
// ---------------------------------------------------------------------------

const collectTools = (
  refs: ReadonlyArray<unknown>,
): ReadonlyArray<BrandedTool> =>
  refs
    .filter(isBrandedTool)
    .map(
      (ref): BrandedTool => ({
        [AgentToolBrand]: true,
        tool: (ref as any).tool ?? (ref as any).constructor?.tool,
        handler: (ref as any).handler ?? (ref as any).constructor?.handler,
      }),
    )
    .filter((t) => t.tool !== undefined && t.handler !== undefined);

const buildAgentClass = (
  id: string,
  template: TemplateStringsArray,
  refs: ReadonlyArray<unknown>,
) => {
  const systemPrompt = renderTemplate(template, refs);
  const tools = collectTools(refs);

  const toolkit =
    tools.length > 0 ? Toolkit.make(...tools.map((t) => t.tool)) : undefined;
  const handlerLayer = toolkit
    ? toolkit.toLayer(
        Object.fromEntries(tools.map((t) => [t.tool.name, t.handler])) as any,
      )
    : undefined;

  class AgentBase {
    static readonly id = id;
    static readonly systemPrompt = systemPrompt;
    static readonly tools = tools;
    static readonly toolkit = toolkit;
    static readonly handlerLayer = handlerLayer;

    static streamText(input: { readonly prompt: Prompt.RawInput }) {
      const userPrompt = Prompt.make(input.prompt);
      const fullPrompt = Prompt.setSystem(userPrompt, systemPrompt);

      if (toolkit === undefined || handlerLayer === undefined) {
        return LanguageModel.streamText({ prompt: fullPrompt });
      }

      return LanguageModel.streamText({
        prompt: fullPrompt,
        toolkit,
      }).pipe(Stream.provide(handlerLayer));
    }

    static generateText(input: { readonly prompt: Prompt.RawInput }) {
      const userPrompt = Prompt.make(input.prompt);
      const fullPrompt = Prompt.setSystem(userPrompt, systemPrompt);

      if (toolkit === undefined || handlerLayer === undefined) {
        return LanguageModel.generateText({ prompt: fullPrompt });
      }

      return LanguageModel.generateText({
        prompt: fullPrompt,
        toolkit,
      }).pipe(Effect.provide(handlerLayer));
    }
  }
  return AgentBase;
};

const agentInlineForm =
  (id: string) =>
  (template: TemplateStringsArray, ...refs: ReadonlyArray<unknown>) =>
    buildAgentClass(id, template, refs);

const agentTagForm =
  () =>
  (id: string) =>
  (template: TemplateStringsArray, ...refs: ReadonlyArray<unknown>) =>
    buildAgentClass(id, template, refs);

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

const renderTemplate = (
  template: TemplateStringsArray,
  refs: ReadonlyArray<unknown>,
): string => {
  let out = "";
  for (let i = 0; i < template.length; i++) {
    out += template[i];
    if (i < refs.length) out += renderRef(refs[i]);
  }
  return dedent(out);
};

const renderRef = (ref: unknown): string => {
  if (isBrandedTool(ref)) {
    const t = (ref as any).tool ?? (ref as any).constructor?.tool;
    return t ? `<tool name="${t.name}"/>` : "<tool/>";
  }
  if (isSchemaStruct(ref)) {
    const keys = Object.keys(ref as Record<string, unknown>);
    return `{ ${keys.map((k) => `"${k}": <${k}>`).join(", ")} }`;
  }
  return String(ref);
};

const isSchemaStruct = (x: unknown): x is Record<string, Schema.Top> => {
  if (typeof x !== "object" || x === null) return false;
  if (isBrandedTool(x)) return false;
  const proto = Object.getPrototypeOf(x);
  if (proto !== Object.prototype && proto !== null) return false;
  const values = Object.values(x as Record<string, unknown>);
  if (values.length === 0) return false;
  return values.every(
    (v) => typeof v === "object" && v !== null && ("ast" in v || "_tag" in v),
  );
};

const mergeStructs = (
  structs: ReadonlyArray<Record<string, Schema.Top>>,
): Schema.Struct<Record<string, Schema.Top>> | undefined => {
  if (structs.length === 0) return undefined;
  const merged: Record<string, Schema.Top> = {};
  for (const s of structs) Object.assign(merged, s);
  return S.Struct(merged) as any;
};

const dedent = (s: string): string => {
  const lines = s.split("\n");
  while (lines.length && lines[0].trim() === "") lines.shift();
  while (lines.length && lines[lines.length - 1].trim() === "") lines.pop();
  const indents = lines
    .filter((l) => l.trim() !== "")
    .map((l) => l.match(/^[ \t]*/)![0].length);
  const min = indents.length ? Math.min(...indents) : 0;
  return lines.map((l) => l.slice(min)).join("\n");
};

// ---------------------------------------------------------------------------
// Examples
// ---------------------------------------------------------------------------

export const Bucket = Cloudflare.R2Bucket("Bucket");

// Option 1 - inline effect
export class CreateObject extends Tool("CreateObject")`
  Create an object in the bucket.
  Return ${{ key: S.String }}
`(
  Effect.gen(function* () {
    const bucket = yield* Cloudflare.R2Bucket.bind(Bucket);
    return Effect.fnUntraced(function* (message) {
      return yield* bucket.put(`${message}.txt`, "data");
    });
  }),
) {}

// Option 2 - tag + layer
export class CreateObjectTag extends Tool<CreateObject>()("CreateObject")`
  Create an object in the bucket.
  Return ${{ key: S.String }}
` {}

export default CreateObjectTag.make(
  Effect.gen(function* () {
    const bucket = yield* Cloudflare.R2Bucket.bind(Bucket);
    return Effect.fnUntraced(function* (message) {
      return yield* bucket.put(`${message}.txt`, "data");
    });
  }),
);

import { OpenAiClient, OpenAiLanguageModel } from "@effect/ai-openai";
import { Config, Layer } from "effect";
import { FetchHttpClient } from "effect/unstable/http";

const OpenAiClientLayer = OpenAiClient.layerConfig({
  apiKey: Config.redacted("OPENAI_API_KEY"),
}).pipe(Layer.provide(FetchHttpClient.layer));

const OpenAiModel = OpenAiLanguageModel.model("gpt-5.2").pipe(
  Layer.provide(OpenAiClientLayer),
);

export class Blogger extends Agent<Blogger>()("Blogger")`
  When asked to, use the ${CreateObject} tool to echo a message back to the user.
` {}

export class BloggerObject extends Cloudflare.DurableObjectNamespace<BloggerObject>()(
  "BloggerObject",
  Effect.gen(function* () {
    const blogger = yield* Blogger;

    // namespace
    return Effect.gen(function* () {
      return {
        fetch: Effect.gen(function* () {
          const response = blogger.streamText({ prompt: "Hello World" }).pipe(
            Stream.flatMap((part) =>
              part.type === "text-delta"
                ? Stream.fromArray([new TextEncoder().encode(part.delta)])
                : Stream.empty,
            ),

            Stream.provide(OpenAiModel),
          );
          // request
          return HttpServerResponse.stream(response, { status: 200 });
        }),
      };
    });
  }),
) {}

export class Api extends Cloudflare.Worker<Api>()(
  "Api",
  {
    main: import.meta.filename,
  },
  Effect.gen(function* () {
    // const blogger = yield* Blogger;
    return {
      fetch: Effect.gen(function* () {
        return HttpServerResponse.text("Hello World", { status: 200 });
      }),
    };
  }),
) {}

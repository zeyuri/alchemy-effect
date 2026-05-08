import * as Cloudflare from "@/Cloudflare/index.ts";
import * as Duration from "effect/Duration";
import * as Effect from "effect/Effect";
import * as Stream from "effect/Stream";
import { LanguageModel as AiLanguageModel } from "effect/unstable/ai";
import { HttpServerRequest } from "effect/unstable/http/HttpServerRequest";
import * as HttpServerResponse from "effect/unstable/http/HttpServerResponse";
import ChatAgent from "./ChatAgent.ts";
import { Gateway } from "./Gateway.ts";

const MODEL = "@cf/meta/llama-3.1-8b-instruct";

export default class LanguageModelTestWorker extends Cloudflare.Worker<LanguageModelTestWorker>()(
  "LanguageModelTestWorker",
  {
    main: import.meta.filename,
    subdomain: { enabled: true, previewsEnabled: false },
    compatibility: { date: "2024-09-23", flags: ["nodejs_compat"] },
  },
  Effect.gen(function* () {
    const aiGateway = yield* Cloudflare.AiGateway.bind(Gateway);
    const languageModel = Cloudflare.AiGatewayLanguageModel.layer({
      client: aiGateway,
      model: MODEL,
      parameters: { temperature: 0.7, maxTokens: 1024 },
    });
    const chatAgents = yield* ChatAgent;

    return {
      fetch: Effect.gen(function* () {
        const request = yield* HttpServerRequest;
        const env = yield* Cloudflare.WorkerEnvironment;
        const url = new URL(request.url, "http://worker");
        const prompt =
          url.searchParams.get("prompt") ??
          "Say the single word 'pong' and nothing else.";

        if (url.pathname === "/chat") {
          const id = url.searchParams.get("id") ?? "default";
          return yield* chatAgents
            .getByName(id)
            .send(prompt)
            .pipe(
              Effect.flatMap((result) => HttpServerResponse.json(result)),
              Effect.catchCause((cause) =>
                HttpServerResponse.json(
                  { error: String(cause) },
                  { status: 500 },
                ),
              ),
            );
        }

        if (url.pathname === "/generate") {
          const response = yield* AiLanguageModel.generateText({ prompt }).pipe(
            Effect.orDie,
          );
          return yield* HttpServerResponse.json({
            text: response.text,
            finishReason: response.finishReason,
            usage: {
              inputTokens: response.usage.inputTokens.total,
              outputTokens: response.usage.outputTokens.total,
            },
          });
        }

        if (url.pathname === "/test-stream") {
          // Synthetic stream: 5 chunks with 200ms gaps. If the client sees
          // them at staggered timestamps, worker→edge streaming works.
          // If they all land at once, output is buffered downstream.
          const encoder = new TextEncoder();
          const body = Stream.range(0, 5).pipe(
            Stream.mapEffect((i) =>
              Effect.sleep(Duration.millis(200)).pipe(
                Effect.as(encoder.encode(`data: chunk-${i}\n\n`)),
              ),
            ),
          );
          return HttpServerResponse.stream(body, {
            headers: { "content-type": "text/event-stream" },
          });
        }

        if (url.pathname === "/stream") {
          const encoder = new TextEncoder();
          const body = AiLanguageModel.streamText({ prompt }).pipe(
            Stream.map((part) =>
              encoder.encode(`data: ${JSON.stringify(part)}\n\n`),
            ),
            Stream.provide(languageModel),
            Stream.provideService(Cloudflare.WorkerEnvironment, env),
          );
          return HttpServerResponse.stream(body, {
            headers: { "content-type": "text/event-stream" },
          });
        }

        return HttpServerResponse.text("ok");
      }).pipe(Effect.provide(languageModel)),
    };
  }).pipe(Effect.provide(Cloudflare.AiGatewayBindingLive)),
) {}

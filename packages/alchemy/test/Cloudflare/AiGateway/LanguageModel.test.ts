import * as Cloudflare from "@/Cloudflare";
import * as Alchemy from "@/index.ts";
import * as Test from "@/Test/Vitest";
import { expect } from "@effect/vitest";
import * as Effect from "effect/Effect";
import { MinimumLogLevel } from "effect/References";
import * as Schedule from "effect/Schedule";
import * as Stream from "effect/Stream";
import * as Sse from "effect/unstable/encoding/Sse";
import * as HttpClient from "effect/unstable/http/HttpClient";
import { Gateway } from "./fixtures/Gateway.ts";
import LanguageModelTestWorker from "./fixtures/LanguageModelWorker.ts";

const { test, beforeAll, afterAll, deploy, destroy } = Test.make({
  providers: Cloudflare.providers(),
});

const logLevel = Effect.provideService(
  MinimumLogLevel,
  process.env.DEBUG ? "Debug" : "Info",
);

const Stack = Alchemy.Stack(
  "AiGatewayLanguageModelStack",
  {
    providers: Cloudflare.providers(),
    state: Cloudflare.state(),
  },
  Effect.gen(function* () {
    const worker = yield* LanguageModelTestWorker;
    const gateway = yield* Gateway;
    return {
      gatewayId: gateway.gatewayId,
      url: worker.url.as<string>(),
    };
  }),
);

const stack = beforeAll(deploy(Stack));
afterAll.skipIf(!!process.env.NO_DESTROY)(destroy(Stack));

test(
  "deployed worker generates text via AiGateway-backed LanguageModel",
  Effect.gen(function* () {
    const out = yield* stack;
    const client = yield* HttpClient.HttpClient;

    const res = yield* client
      .get(`${out.url}/generate?prompt=${encodeURIComponent("Say pong.")}`)
      .pipe(
        Effect.retry({
          schedule: Schedule.exponential("500 millis"),
          times: 10,
        }),
      );
    expect(res.status).toBe(200);

    const body = (yield* res.json) as {
      text: string;
      finishReason: string;
      usage: {
        inputTokens: number | undefined;
        outputTokens: number | undefined;
      };
    };

    expect(typeof body.text).toBe("string");
    expect(body.text.length).toBeGreaterThan(0);
    expect(body.finishReason).not.toBe("error");
    expect(body.usage.outputTokens).toBeGreaterThan(0);
  }).pipe(logLevel),
  { timeout: 180_000 },
);

test(
  "deployed worker streams text via AiGateway-backed LanguageModel",
  Effect.gen(function* () {
    const out = yield* stack;
    const client = yield* HttpClient.HttpClient;

    const res = yield* client
      .get(`${out.url}/stream?prompt=${encodeURIComponent("Say pong.")}`)
      .pipe(
        Effect.retry({
          schedule: Schedule.exponential("500 millis"),
          times: 10,
        }),
      );
    expect(res.status).toBe(200);

    const sse = yield* res.text;
    const parts = sse
      .split("\n\n")
      .map((frame) => frame.replace(/^data:\s*/, "").trim())
      .filter((line) => line.length > 0)
      .map((line) => JSON.parse(line) as { type: string; delta?: string });

    const text = parts
      .filter((p) => p.type === "text-delta")
      .map((p) => p.delta ?? "")
      .join("");
    const finish = parts.find((p) => p.type === "finish");

    expect(parts.length).toBeGreaterThan(0);
    expect(text.length).toBeGreaterThan(0);
    expect(finish).toBeDefined();
  }).pipe(logLevel),
  { timeout: 180_000 },
);

test(
  "persisted chat survives across DO invocations",
  Effect.gen(function* () {
    const out = yield* stack;
    const client = yield* HttpClient.HttpClient;
    const id = `test-${Date.now()}`;

    const r1 = yield* client
      .get(
        `${out.url}/chat?id=${id}&prompt=${encodeURIComponent("My name is Sam. Remember it.")}`,
      )
      .pipe(
        Effect.retry({
          schedule: Schedule.exponential("500 millis"),
          times: 10,
        }),
      );
    if (r1.status !== 200) {
      console.error("turn1 error body:", yield* r1.text);
    }
    expect(r1.status).toBe(200);
    const b1 = (yield* r1.json) as { text: string; turns: number };
    expect(b1.turns).toBeGreaterThanOrEqual(2);

    const r2 = yield* client
      .get(
        `${out.url}/chat?id=${id}&prompt=${encodeURIComponent("What is my name? Answer with just the name.")}`,
      )
      .pipe(
        Effect.retry({
          schedule: Schedule.exponential("500 millis"),
          times: 10,
        }),
      );
    expect(r2.status).toBe(200);
    const b2 = (yield* r2.json) as { text: string; turns: number };

    expect(b2.text.toLowerCase()).toContain("sam");
    expect(b2.turns).toBeGreaterThanOrEqual(4);
  }).pipe(logLevel),
  { timeout: 240_000 },
);

test(
  "streams Effect-native parts and prints them live",
  Effect.gen(function* () {
    const out = yield* stack;
    const client = yield* HttpClient.HttpClient;

    const res = yield* client
      .get(
        `${out.url}/stream?prompt=${encodeURIComponent("Write a short haiku about Effect TS.")}`,
      )
      .pipe(
        Effect.retry({
          schedule: Schedule.exponential("500 millis"),
          times: 10,
        }),
      );
    expect(res.status).toBe(200);

    // Write to fd 2 (stderr) with fs.writeSync to bypass vitest's stdout
    // capture and Node's stream buffering — chunks land in the terminal as
    // soon as they arrive from the network.
    const fs = yield* Effect.promise(() => import("node:fs"));
    const print = (s: string) => fs.writeSync(2, s);

    print("\n--- live stream begin ---\n");
    let collected = "";

    yield* res.stream.pipe(
      Stream.orDie,
      Stream.decodeText(),
      Stream.pipeThroughChannel(Sse.decode<never, unknown>()),
      Stream.runForEach((event) =>
        Effect.sync(() => {
          const part = JSON.parse(event.data) as {
            type: string;
            delta?: string;
            id?: string;
            reason?: string;
          };
          switch (part.type) {
            case "text-start":
              print(`[text-start id=${part.id}]\n`);
              break;
            case "text-delta":
              print(part.delta ?? "");
              collected += part.delta ?? "";
              break;
            case "text-end":
              print(`\n[text-end id=${part.id}]\n`);
              break;
            case "finish":
              print(`[finish reason=${part.reason}]\n`);
              break;
            default:
              print(`[${part.type}]\n`);
          }
        }),
      ),
    );

    print("--- live stream end ---\n");
    expect(collected.length).toBeGreaterThan(0);
  }).pipe(logLevel),
  { timeout: 180_000 },
);

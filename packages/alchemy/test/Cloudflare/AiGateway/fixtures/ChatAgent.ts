import * as Cloudflare from "@/Cloudflare/index.ts";
import * as Effect from "effect/Effect";
import * as Ref from "effect/Ref";
import { Chat, LanguageModel } from "effect/unstable/ai";
import { Gateway } from "./Gateway.ts";

const MODEL = "@cf/meta/llama-3.1-8b-instruct";

/**
 * A Durable Object that owns a single stateful chat session.
 *
 * - **Init phase** binds the AI Gateway client and constructs the
 *   `LanguageModel` layer.
 * - **Instance phase** wires `Chat.Persistence` on top of the DO's own
 *   `state.storage` (via `DurableObjectBackingPersistence`) and resolves a
 *   single persisted chat for this DO instance.
 *
 * Because the chat lives inside the DO, the conversation history survives
 * across DO invocations — `state.storage` is the persistence layer.
 */
export default class ChatAgent extends Cloudflare.DurableObjectNamespace<ChatAgent>()(
  "ChatAgent",
  Effect.gen(function* () {
    const aiGateway = yield* Cloudflare.AiGateway.bind(Gateway);
    const languageModel = Cloudflare.AiGatewayLanguageModel.layer({
      client: aiGateway,
      model: MODEL,
      parameters: { temperature: 0, maxTokens: 64 },
    });

    return Effect.gen(function* () {
      const persistence = yield* Chat.Persistence;
      const chat = yield* persistence.getOrCreate("session");
      // Resolve the LanguageModel service in the instance phase (where
      // `WorkerEnvironment` is in context) and capture it so each `send`
      // RPC call can re-provide it without rebuilding the layer (RPC
      // invocations don't have `WorkerEnvironment` in context).
      const lm = yield* LanguageModel.LanguageModel;

      return {
        send: (prompt: string) =>
          Effect.gen(function* () {
            const response = yield* chat.generateText({ prompt });
            const history = yield* Ref.get(chat.history);
            return { text: response.text, turns: history.content.length };
          }).pipe(
            Effect.provideService(LanguageModel.LanguageModel, lm),
            Effect.orDie,
          ),
      };
    }).pipe(
      Effect.provide(languageModel),
      Effect.provide(Chat.layerPersisted({ storeId: "chat" })),
      Effect.provide(Cloudflare.DurableObjectBackingPersistence.layer),
      Effect.orDie,
    );
  }),
) {}

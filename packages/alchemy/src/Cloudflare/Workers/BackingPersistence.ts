import * as Arr from "effect/Array";
import * as Effect from "effect/Effect";
import * as Layer from "effect/Layer";
import {
  BackingPersistence,
  PersistenceError,
  type BackingPersistenceStore,
} from "effect/unstable/persistence/Persistence";
import { DurableObjectState } from "./DurableObjectState.ts";

/**
 * A `BackingPersistence` layer backed by a Durable Object's `state.storage`.
 *
 * Multiple `storeId`s can coexist within a single Durable Object — each store's
 * keys are namespaced with a `${storeId}:` prefix.
 *
 * @remarks
 * TTL is currently ignored; Durable Object storage has no native TTL. If you
 * need expiry, layer a TTL-aware backing store on top.
 */
export const layer = Layer.effect(BackingPersistence)(
  Effect.gen(function* () {
    const state = yield* DurableObjectState;
    const storage = state.storage;

    const wrapErr = (op: string, key?: string) => (cause: unknown) =>
      new PersistenceError({
        message: `Failed to ${op}${key !== undefined ? ` key ${key}` : ""} in DurableObject storage`,
        cause,
      });

    return BackingPersistence.of({
      make: (storeId) =>
        Effect.sync(() => {
          const prefixed = (k: string) => `${storeId}:${k}`;
          return {
            get: (key) =>
              storage
                .get<object>(prefixed(key))
                .pipe(Effect.mapError(wrapErr("get", key))),
            getMany: (keys) =>
              storage.get<object>(keys.map(prefixed)).pipe(
                Effect.mapError(wrapErr("getMany")),
                Effect.map(
                  (map) =>
                    keys.map((k) => map.get(prefixed(k))) as Arr.NonEmptyArray<
                      object | undefined
                    >,
                ),
              ),
            set: (key, value, _ttl) =>
              storage
                .put(prefixed(key), value)
                .pipe(Effect.mapError(wrapErr("set", key))),
            setMany: (entries) =>
              storage
                .put(
                  Object.fromEntries(entries.map(([k, v]) => [prefixed(k), v])),
                )
                .pipe(Effect.mapError(wrapErr("setMany"))),
            remove: (key) =>
              storage
                .delete(prefixed(key))
                .pipe(Effect.asVoid, Effect.mapError(wrapErr("remove", key))),
            clear: storage.list({ prefix: `${storeId}:` }).pipe(
              Effect.flatMap((map) => {
                const ks = [...map.keys()];
                if (ks.length === 0) return Effect.void;
                return Effect.asVoid(storage.delete(ks));
              }),
              Effect.mapError(wrapErr("clear")),
            ),
          } satisfies BackingPersistenceStore;
        }),
    });
  }),
);

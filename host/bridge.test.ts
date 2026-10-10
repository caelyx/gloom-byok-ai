import { afterEach, describe, expect, test } from "bun:test";
import {
  AI_RUNNER_CAPABILITY_ID,
  setCapabilityStreamClient,
  type CapabilityStreamSubscription,
} from "gloomberb/capabilities";
import type { AiRuntimeCatalog } from "../runner";
import { createBridgeAiRunHost } from "./bridge";

const catalog: AiRuntimeCatalog = { providers: [], accounts: [], models: [] };

function streamClient(respond: (subscription: CapabilityStreamSubscription) => void) {
  const subscriptions: CapabilityStreamSubscription[] = [];
  let unsubscribed = 0;
  setCapabilityStreamClient({
    invoke: async () => {
      throw new Error("API keys must not be sent with invoke.");
    },
    subscribe(subscription) {
      subscriptions.push(subscription);
      queueMicrotask(() => respond(subscription));
      return () => {
        unsubscribed += 1;
      };
    },
  });
  return { subscriptions, unsubscribed: () => unsubscribed };
}

afterEach(() => {
  setCapabilityStreamClient(null);
});

describe("desktop bridge API-key saving", () => {
  test("streams the key to the Bun process and ends the stream once saved", async () => {
    const client = streamClient((subscription) => {
      subscription.onEvent({ kind: "account-connected", catalog });
    });

    expect(await createBridgeAiRunHost().saveApiKey!("google", "AIza-secret")).toEqual(catalog);
    expect(client.subscriptions.map(({ capabilityId, operationId, payload }) => ({
      capabilityId,
      operationId,
      payload,
    }))).toEqual([{
      capabilityId: AI_RUNNER_CAPABILITY_ID,
      operationId: "saveProviderApiKey",
      payload: { providerId: "google", apiKey: "AIza-secret" },
    }]);
    expect(client.unsubscribed()).toBe(1);
  });

  test("rejects with the Bun process's error", async () => {
    streamClient((subscription) => {
      subscription.onEvent({ kind: "account-error", error: "Enter an API key." });
    });

    await expect(createBridgeAiRunHost().saveApiKey!("google", " ")).rejects.toThrow("Enter an API key.");
  });
});

import { describe, expect, test } from "bun:test";
import type { AiRunHost, AiRuntimeCatalog } from "../runner";
import { createAiRunnerCapability } from "./capability";

const catalog: AiRuntimeCatalog = {
  providers: [{
    providerId: "google",
    label: "Google Gemini",
    status: "ready",
    outputModes: ["plain", "structured", "screener"],
  }],
  accounts: [],
  models: [],
};

function save(capability: ReturnType<typeof createAiRunnerCapability>, payload: unknown) {
  return new Promise((resolve, reject) => {
    void Promise.resolve(capability.operations.saveProviderApiKey!.subscribe!(payload, (event: any) => {
      if (event.kind === "account-connected") resolve(event.catalog);
      else if (event.kind === "account-error") reject(new Error(event.error));
    }, {} as never)).catch(reject);
  });
}

function capabilityWith(host: Partial<AiRunHost>) {
  return createAiRunnerCapability(async () => ({
    run: () => ({ done: Promise.resolve(""), cancel() {} }),
    getCatalog: async () => catalog,
    ...host,
  }));
}

describe("AI runner capability API-key entry", () => {
  test("forwards a typed key to the Bun host only as a renderer stream", async () => {
    const saved: Array<[string, string]> = [];
    const capability = capabilityWith({
      async saveApiKey(providerId, apiKey) {
        saved.push([providerId, apiKey]);
        return catalog;
      },
    });
    const operation = capability.operations.saveProviderApiKey!;

    // Remote control reaches capabilities through invoke only, so an agent
    // with the remote tool must not find a handler here.
    expect(operation).toMatchObject({ kind: "stream", rendererSafe: true });
    expect(operation.handler).toBeUndefined();
    expect(await save(capability, { providerId: "google", apiKey: "AIza-secret" })).toEqual(catalog);
    expect(saved).toEqual([["google", "AIza-secret"]]);
  });

  test("rejects a missing key and an unknown provider before reaching the host", async () => {
    const saved: string[] = [];
    const capability = capabilityWith({
      async saveApiKey(providerId) {
        saved.push(providerId);
        return catalog;
      },
    });
    await expect(save(capability, { providerId: "google", apiKey: "" })).rejects.toThrow("API key is required");
    await expect(save(capability, { providerId: "openrouter", apiKey: "sk-or" })).rejects.toThrow("Unknown AI provider");
    expect(saved).toEqual([]);
  });
});

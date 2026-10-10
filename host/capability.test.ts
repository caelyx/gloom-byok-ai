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

function capabilityWith(host: Partial<AiRunHost>) {
  return createAiRunnerCapability(async () => ({
    run: () => ({ done: Promise.resolve(""), cancel() {} }),
    getCatalog: async () => catalog,
    ...host,
  }));
}

describe("AI runner capability API-key entry", () => {
  test("forwards a typed key to the Bun host as a renderer-safe action", async () => {
    const saved: Array<[string, string]> = [];
    const capability = capabilityWith({
      async saveApiKey(providerId, apiKey) {
        saved.push([providerId, apiKey]);
        return catalog;
      },
    });
    const operation = capability.operations.saveProviderApiKey!;

    expect(operation).toMatchObject({ kind: "action", rendererSafe: true });
    expect(await operation.handler!({ providerId: "google", apiKey: "AIza-secret" }, {} as never))
      .toEqual(catalog);
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
    const handler = capability.operations.saveProviderApiKey!.handler!;

    await expect(handler({ providerId: "google", apiKey: "" }, {} as never)).rejects.toThrow("API key is required");
    await expect(handler({ providerId: "openrouter", apiKey: "sk-or" }, {} as never)).rejects.toThrow("Unknown AI provider");
    expect(saved).toEqual([]);
  });
});

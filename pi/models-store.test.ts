import { afterEach, describe, expect, test } from "bun:test";
import { fauxProvider } from "@earendil-works/pi-ai";
import { mkdtemp, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { PiFileModelsStore } from "./models-store";

const tempDirs: string[] = [];

async function tempDataDir(): Promise<string> {
  const path = await mkdtemp(join(tmpdir(), "gloomberb-pi-models-"));
  tempDirs.push(path);
  return path;
}

afterEach(async () => {
  await Promise.all(tempDirs.splice(0).map((path) => rm(path, { recursive: true, force: true })));
});

describe("PiFileModelsStore", () => {
  test("persists provider-scoped model catalogs without sharing mutable values", async () => {
    const dataDir = await tempDataDir();
    const store = new PiFileModelsStore(dataDir);
    const faux = fauxProvider({ provider: "catalog-provider", models: [{ id: "catalog-model" }] });
    const entry = { models: faux.models, checkedAt: 123 };

    await store.write("catalog-provider", entry);
    const first = await store.read("catalog-provider");
    expect(first).toMatchObject({ checkedAt: 123, models: [{ id: "catalog-model", provider: "catalog-provider" }] });
    if (first?.models[0]) first.models[0].name = "mutated";

    expect((await new PiFileModelsStore(dataDir).read("catalog-provider"))?.models[0]?.name).not.toBe("mutated");
  });

  test("rejects a catalog stored under the wrong provider", async () => {
    const dataDir = await tempDataDir();
    const store = new PiFileModelsStore(dataDir);
    const faux = fauxProvider({ provider: "actual-provider" });

    await expect(store.write("other-provider", { models: faux.models })).rejects.toThrow(
      "does not match other-provider",
    );
  });

  test("accepts Pi 1.x image and classifier catalog entries", async () => {
    const dataDir = await tempDataDir();
    const store = new PiFileModelsStore(dataDir);
    const base = {
      name: "Typed",
      provider: "typed-provider",
      baseUrl: "https://example.test",
      input: ["text"],
      cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 },
    } as const;
    const entry = {
      models: [
        { ...base, id: "image-model", api: "openrouter-images", type: "image", output: ["image"] },
        { ...base, id: "classifier-model", api: "typesafe-system-one", type: "classifier", contextWindow: 8_192 },
      ],
      etag: '"abc"',
    } as const;

    await store.write("typed-provider", structuredClone(entry) as never);
    expect(await new PiFileModelsStore(dataDir).read("typed-provider")).toEqual(structuredClone(entry) as never);
  });
});

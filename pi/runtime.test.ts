import { describe, expect, test } from "bun:test";
import {
  InMemoryCredentialStore,
  Type,
  createModels,
  createProvider,
  envApiKeyAuth,
  fauxAssistantMessage,
  fauxProvider,
  fauxToolCall,
} from "@earendil-works/pi-ai";
import { anthropicProvider } from "@earendil-works/pi-ai/providers/anthropic";
import { githubCopilotProvider } from "@earendil-works/pi-ai/providers/github-copilot";
import { googleProvider } from "@earendil-works/pi-ai/providers/google";
import { openrouterProvider } from "@earendil-works/pi-ai/providers/openrouter";
import {
  PiAiRuntime,
  PiRunCancelledError,
  PiRuntimeError,
  defaultReasoning,
} from "./runtime";

function createFauxRuntime(options: { tokensPerSecond?: number } = {}) {
  const faux = fauxProvider({
    provider: "anthropic",
    models: [
      { id: "claude-fable-5-1", name: "Fable" },
      { id: "claude-sonnet-5-5", name: "Sonnet" },
    ],
    tokensPerSecond: options.tokensPerSecond,
  });
  const models = createModels({ credentials: new InMemoryCredentialStore() });
  models.setProvider(faux.provider);
  return { faux, runtime: new PiAiRuntime({ models }) };
}

describe("PiAiRuntime", () => {
  test("exposes a serializable canonical connected catalog", async () => {
    const { runtime } = createFauxRuntime();
    const catalog = await runtime.getCatalog();

    expect(catalog.providers).toHaveLength(1);
    expect(catalog.providers[0]).toMatchObject({
      id: "anthropic",
      label: "Claude",
      connection: {
        state: "connected",
        type: "api_key",
        origin: "external",
        disconnectable: false,
      },
    });
    expect(catalog.providers[0]).not.toHaveProperty("uiId");
    expect(catalog.providers[0]?.models.map((model) => model.id)).toEqual([
      "claude-fable-5-1",
      "claude-sonnet-5-5",
    ]);
  });

  test("uses the provider default for a blank model and streams cumulative text", async () => {
    const { faux, runtime } = createFauxRuntime();
    let requestedModel = "";
    faux.setResponses([
      (context, _options, _state, model) => {
        requestedModel = model.id;
        expect(context.messages.at(-1)).toMatchObject({ role: "user", content: "hello" });
        return fauxAssistantMessage("world");
      },
    ]);
    const chunks: string[] = [];

    const result = await runtime.runText({
      providerId: "anthropic",
      modelId: "",
      prompt: "hello",
      onChunk: (chunk) => chunks.push(chunk),
    }).done;

    expect(requestedModel).toBe("claude-sonnet-5-5");
    expect(result).toBe("world");
    expect(chunks.at(-1)).toBe("world");
  });

  test("rejects legacy provider aliases at the Pi runtime boundary", async () => {
    const { runtime } = createFauxRuntime();

    await expect(runtime.getProviderSummary("claude"))
      .rejects.toMatchObject({ code: "provider_not_found" });
  });

  test("rejects an injected legacy provider instead of publishing a mismatched canonical entry", async () => {
    const legacy = fauxProvider({
      provider: "claude",
      models: [{ id: "claude-sonnet-5-5", name: "Sonnet" }],
    });
    const models = createModels({ credentials: new InMemoryCredentialStore() });
    models.setProvider(legacy.provider);
    const runtime = new PiAiRuntime({ models });

    await expect(runtime.getCatalog()).rejects.toMatchObject({
      code: "provider_not_found",
      message: "Unsupported AI provider: claude",
    });
  });

  test("bridges OAuth events and prompts without returning credentials", async () => {
    const credentials = new InMemoryCredentialStore();
    const faux = fauxProvider({ provider: "anthropic", models: [{ id: "claude-sonnet-5-5" }] });
    const provider = createProvider({
      id: "anthropic",
      name: "Anthropic",
      auth: {
        oauth: {
          name: "Claude Pro/Max",
          login: async (interaction) => {
            interaction.notify({ type: "auth_url", url: "https://example.test/login" });
            const code = await interaction.prompt({ type: "text", message: "Paste code" });
            return { type: "oauth", access: `access-${code}`, refresh: "refresh-secret", expires: Date.now() + 60_000 };
          },
          refresh: async (credential) => credential,
          toAuth: async (credential) => ({ apiKey: credential.access }),
        },
      },
      models: faux.models,
      api: { stream: faux.provider.stream, streamSimple: faux.provider.streamSimple },
    });
    const models = createModels({ credentials });
    models.setProvider(provider);
    const runtime = new PiAiRuntime({ models });
    const events: unknown[] = [];
    const prompts: unknown[] = [];

    const connected = await runtime.login(
      { providerId: "anthropic", type: "oauth" },
      {
        notify: (event) => events.push(event),
        prompt: async (prompt) => {
          prompts.push(prompt);
          return "login-code";
        },
      },
    );

    expect(events).toEqual([{ type: "auth_url", url: "https://example.test/login" }]);
    expect(prompts).toEqual([{ type: "text", message: "Paste code" }]);
    expect(connected.connection).toEqual({
      state: "connected",
      type: "oauth",
      source: "OAuth",
      origin: "stored",
      disconnectable: true,
    });
    expect(JSON.stringify(connected)).not.toContain("login-code");
    expect((await runtime.logout("anthropic")).connection).toEqual({ state: "not_connected" });
  });

  test("stores a pasted API key in place of an OAuth sign-in without echoing it", async () => {
    const credentials = new InMemoryCredentialStore();
    await credentials.modify("anthropic", async () => ({
      type: "oauth",
      access: "subscription-access",
      refresh: "subscription-refresh",
      expires: Date.now() + 60_000,
    }));
    const faux = fauxProvider({ provider: "anthropic", models: [{ id: "claude-sonnet-5-5" }] });
    const provider = createProvider({
      id: "anthropic",
      name: "Anthropic",
      auth: {
        apiKey: envApiKeyAuth("Anthropic API key", ["GLOOM_TEST_UNSET_ANTHROPIC_KEY"]),
        oauth: {
          name: "Claude Pro/Max",
          login: async () => { throw new Error("unused"); },
          refresh: async (credential) => credential,
          toAuth: async (credential) => ({ apiKey: credential.access }),
        },
      },
      models: faux.models,
      api: { stream: faux.provider.stream, streamSimple: faux.provider.streamSimple },
    });
    const models = createModels({ credentials });
    models.setProvider(provider);
    const runtime = new PiAiRuntime({ models });

    const connected = await runtime.saveApiKey("anthropic", "  sk-ant-api03-console-key \n");

    expect(await credentials.read("anthropic")).toEqual({ type: "api_key", key: "sk-ant-api03-console-key" });
    expect(connected.connection).toEqual({
      state: "connected",
      type: "api_key",
      source: "stored credential",
      origin: "stored",
      disconnectable: true,
    });
    expect(JSON.stringify(connected)).not.toContain("sk-ant-api03");
  });

  test("refuses a blank API key and providers without API-key entry", async () => {
    const { runtime } = createFauxRuntime();

    await expect(runtime.saveApiKey("anthropic", "   ")).rejects.toMatchObject({
      code: "login_failed",
      message: "Enter an API key.",
    });
    await expect(runtime.saveApiKey("anthropic", "sk-test")).rejects.toMatchObject({
      code: "login_failed",
    });
  });

  test("does not report an expired OAuth credential as connected when refresh fails", async () => {
    const credentials = new InMemoryCredentialStore();
    await credentials.modify("anthropic", async () => ({
      type: "oauth",
      access: "expired-access",
      refresh: "expired-refresh",
      expires: Date.now() - 1,
    }));
    const faux = fauxProvider({ provider: "anthropic", models: [{ id: "claude-sonnet-5-5" }] });
    const provider = createProvider({
      id: "anthropic",
      name: "Anthropic",
      auth: {
        oauth: {
          name: "Claude Pro/Max",
          login: async () => { throw new Error("unused"); },
          refresh: async () => { throw new Error("refresh rejected"); },
          toAuth: async (credential) => ({ apiKey: credential.access }),
        },
      },
      models: faux.models,
      api: { stream: faux.provider.stream, streamSimple: faux.provider.streamSimple },
    });
    const models = createModels({ credentials });
    models.setProvider(provider);
    const runtime = new PiAiRuntime({ models });

    expect((await runtime.getProviderSummary("anthropic")).connection).toMatchObject({
      state: "error",
      message: expect.stringContaining("OAuth refresh failed"),
    });
    await expect(runtime.resolveModel({ providerId: "anthropic" }))
      .rejects.toMatchObject({ code: "provider_not_configured" });
  });

  test("normalizes provider failures and cancellation", async () => {
    const failed = createFauxRuntime();
    failed.faux.setResponses([
      fauxAssistantMessage([], {
        stopReason: "error",
        errorMessage: "Authorization: Bearer super-secret-token",
      }),
    ]);
    const failure = failed.runtime.runText({ providerId: "anthropic", prompt: "fail" }).done;
    await expect(failure).rejects.toBeInstanceOf(PiRuntimeError);
    await expect(failure).rejects.not.toThrow("super-secret-token");

    const slow = createFauxRuntime({ tokensPerSecond: 0.01 });
    slow.faux.setResponses([fauxAssistantMessage("This response should be cancelled.")]);
    const run = slow.runtime.runText({ providerId: "anthropic", prompt: "cancel" });
    run.cancel();
    await expect(run.done).rejects.toBeInstanceOf(PiRunCancelledError);
  });

  test("runs typed tools through pi-agent-core and returns the final agent response", async () => {
    const { faux, runtime } = createFauxRuntime();
    faux.setResponses([
      fauxAssistantMessage(fauxToolCall("remote_control", { request: "schema" }), { stopReason: "toolUse" }),
      fauxAssistantMessage("Remote schema loaded."),
    ]);
    const toolCalls: unknown[] = [];
    const chunks: string[] = [];

    const run = runtime.runAgent({
      providerId: "anthropic",
      prompt: "Inspect the app",
      tools: [{
        name: "remote_control",
        label: "Gloomberb remote control",
        description: "Send a request to the running Gloomberb app.",
        parameters: Type.Object({ request: Type.String() }),
        execute: async (_toolCallId, params) => {
          const request = (params as { request: string }).request;
          toolCalls.push({ request });
          return {
            content: [{ type: "text", text: '{"actions":[]}' }],
            details: { request },
          };
        },
      }],
      onChunk: (chunk) => chunks.push(chunk),
    });

    const result = await run.done;
    expect(toolCalls).toEqual([{ request: "schema" }]);
    expect(result.text).toBe("Remote schema loaded.");
    expect(chunks.at(-1)).toBe("Remote schema loaded.");
    expect(result.messages.some((message) => message.role === "toolResult")).toBe(true);
  });
});

describe("saved model selections", () => {
  test("say how to recover when a saved model is no longer offered", async () => {
    const { runtime } = createFauxRuntime();

    await expect(runtime.resolveModel({ providerId: "anthropic", modelId: "claude-opus-4-1" }))
      .rejects.toMatchObject({
        code: "model_not_found",
        message: "Claude does not offer claude-opus-4-1. Choose another model.",
      });
  });
});

describe("default Claude effort", () => {
  test("is medium for adaptive-thinking Claude models and unset elsewhere", () => {
    const claude = new Map(anthropicProvider().getModels().map((model) => [model.id, model]));
    for (const modelId of ["claude-sonnet-5-5", "claude-opus-5-5", "claude-haiku-5-5", "claude-fable-5-1", "claude-opus-4-8"]) {
      expect(defaultReasoning(claude.get(modelId)!)).toBe("medium");
    }
    // Budget-thinking models have no effort; medium would switch thinking on.
    expect(defaultReasoning(claude.get("claude-sonnet-4-5")!)).toBeUndefined();
    expect(defaultReasoning(googleProvider().getModels().find((model) => model.id === "gemini-3.8-flash")!))
      .toBeUndefined();
  });

  test("covers Claude reached through Copilot and OpenRouter", () => {
    const copilot = githubCopilotProvider().getModels();
    const openrouter = openrouterProvider().getModels();

    expect(defaultReasoning(copilot.find((model) => model.id === "claude-sonnet-5.5")!)).toBe("medium");
    expect(defaultReasoning(openrouter.find((model) => model.id === "anthropic/claude-sonnet-5.5")!)).toBe("medium");
    expect(defaultReasoning(copilot.find((model) => model.id === "gpt-6.1-sol")!)).toBeUndefined();
  });

  test("leaves non-Claude models on Anthropic's API alone", () => {
    const [claude] = anthropicProvider().getModels().filter((model) => model.id === "claude-sonnet-5-5");

    expect(defaultReasoning({ ...claude!, id: "kimi-for-coding", provider: "kimi-coding" })).toBeUndefined();
  });

  function adaptiveClaudeRuntime() {
    const faux = fauxProvider({
      provider: "anthropic",
      api: "anthropic-messages",
      models: [{ id: "claude-sonnet-5-5", name: "Sonnet", reasoning: true }],
    });
    // The faux provider takes no compat flags, so mark its model as an
    // adaptive-thinking one the way Pi's Anthropic catalog does.
    for (const model of faux.models) {
      (model as { compat?: unknown }).compat = { forceAdaptiveThinking: true };
    }
    const models = createModels({ credentials: new InMemoryCredentialStore() });
    models.setProvider(faux.provider);
    return { faux, runtime: new PiAiRuntime({ models }) };
  }

  function recordReasoning(faux: ReturnType<typeof adaptiveClaudeRuntime>["faux"], count: number) {
    const levels: unknown[] = [];
    faux.setResponses(Array.from({ length: count }, () => (_context: unknown, options?: { reasoning?: unknown }) => {
      levels.push(options?.reasoning);
      return fauxAssistantMessage("ok");
    }));
    return levels;
  }

  test("reaches plain runs", async () => {
    const { faux, runtime } = adaptiveClaudeRuntime();
    const levels = recordReasoning(faux, 1);

    await runtime.runText({ providerId: "anthropic", modelId: "claude-sonnet-5-5", prompt: "a" }).done;

    expect(levels).toEqual(["medium"]);
  });

  test("gives way to a level a plain run asks for, including off", async () => {
    const { faux, runtime } = adaptiveClaudeRuntime();
    const levels = recordReasoning(faux, 2);

    await runtime.runText({ providerId: "anthropic", modelId: "claude-sonnet-5-5", prompt: "a", reasoning: "high" }).done;
    await runtime.runText({ providerId: "anthropic", modelId: "claude-sonnet-5-5", prompt: "b", reasoning: "off" }).done;

    expect(levels).toEqual(["high", undefined]);
  });

  test("reaches AI Agent threads and gives way to a requested level", async () => {
    const { faux, runtime } = adaptiveClaudeRuntime();
    const levels = recordReasoning(faux, 2);

    await runtime.runAgent({ providerId: "anthropic", modelId: "claude-sonnet-5-5", prompt: "go" }).done;
    await runtime.runAgent({ providerId: "anthropic", modelId: "claude-sonnet-5-5", prompt: "go", thinkingLevel: "off" }).done;

    expect(levels).toEqual(["medium", undefined]);
  });
});

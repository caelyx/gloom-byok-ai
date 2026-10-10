import { describe, expect, test } from "bun:test";
import {
  InMemoryCredentialStore,
  createModels,
  fauxAssistantMessage,
  fauxProvider,
  fauxToolCall,
  getCurrentSystemPrompt,
  getCurrentTools,
} from "@earendil-works/pi-ai";
import { AI_PROVIDER_IDS, type AiProviderId } from "../providers";
import type { AiAgentHistoryMessage } from "../agent-history";
import { parseScreenerResponse } from "../screener/contract";
import type {
  RemoteControlRequest,
  RemoteControlResponse,
} from "gloomberb/remote";
import { createPiAiHost, toAiRuntimeCatalog } from "./host";
import { PiAiRuntime, type PiProviderSummary } from "./runtime";

function createHostFixture(sendRemoteRequest?: (
  request: RemoteControlRequest,
  options: { dataDir: string; appKind?: "tui" | "desktop" },
) => Promise<RemoteControlResponse>) {
  const faux = fauxProvider({
    provider: "anthropic",
    models: [{ id: "claude-sonnet-5-5", name: "Sonnet" }],
  });
  const models = createModels({ credentials: new InMemoryCredentialStore() });
  models.setProvider(faux.provider);
  const host = createPiAiHost({
    appKind: "tui",
    dataDir: "/tmp/gloomberb-pi-host-test",
    runtime: new PiAiRuntime({ models }),
    sendRemoteRequest,
  });
  return { faux, host };
}

function candidates(count: number) {
  return Array.from({ length: count }, (_, index) => ({
    symbol: `T${index}`,
    exchange: "NASDAQ",
    reason: `Candidate ${index}`,
  }));
}

function disconnectedSummary(id: AiProviderId, label: string): PiProviderSummary {
  return {
    id,
    label,
    name: label,
    defaultModelId: "default-model",
    authMethods: [{ type: "oauth", label: `Connect ${label}`, canLogin: true }],
    connection: { state: "not_connected" },
    models: [],
  };
}

describe("Pi AI host screener mode", () => {
  test("rejects an oversized tool submission and returns corrected parser-compatible JSON", async () => {
    const fixture = createHostFixture();
    fixture.faux.setResponses([
      fauxAssistantMessage(fauxToolCall("submit_screener_results", {
        title: "Too many",
        tickers: candidates(26),
      }), { stopReason: "toolUse" }),
      fauxAssistantMessage(fauxToolCall("submit_screener_results", {
        title: "Top names",
        summary: "Validated candidates",
        tickers: candidates(25),
      }), { stopReason: "toolUse" }),
    ]);
    const chunks: string[] = [];

    const output = await fixture.host.run({
      providerId: "anthropic",
      prompt: "Find candidates",
      outputMode: "screener",
      onChunk: (chunk) => chunks.push(chunk),
    }).done;

    expect(JSON.parse(output)).toEqual({
      title: "Top names",
      summary: "Validated candidates",
      tickers: candidates(25),
    });
    expect(parseScreenerResponse(output)).toEqual({
      title: "Top names",
      summary: "Validated candidates",
      tickers: candidates(25),
    });
    expect(chunks).toEqual([output]);
    expect(fixture.faux.state.callCount).toBe(2);
  });

  test("fails closed when the model never calls the submission tool", async () => {
    const fixture = createHostFixture();
    fixture.faux.setResponses([fauxAssistantMessage('{"tickers":[]}')]);

    const run = fixture.host.run({
      providerId: "anthropic",
      prompt: "Find nothing",
      outputMode: "screener",
    });

    await expect(run.done).rejects.toThrow("without submitting structured results");
  });

  test("exposes only read-only market data and submission tools to the screener", async () => {
    const requests: RemoteControlRequest[] = [];
    const fixture = createHostFixture(async (request) => {
      requests.push(request);
      return {
        ok: true,
        data: {
          symbol: "NVDA",
          exchange: "NASDAQ",
          price: 180,
        },
      };
    });
    fixture.faux.setResponses([
      (context) => {
        const tools = getCurrentTools(context.messages);
        expect(tools.map((tool) => tool.name)).toEqual([
          "gloomberb_market_data",
          "submit_screener_results",
        ]);
        const marketDataTool = tools.find((tool) => tool.name === "gloomberb_market_data");
        const toolDefinition = JSON.stringify(marketDataTool);
        expect(toolDefinition).toContain('"quote"');
        expect(toolDefinition).not.toContain("gloomberb_remote");
        expect(toolDefinition).not.toContain("app.openCommandBar");
        expect(toolDefinition).not.toContain('"call"');
        expect(getCurrentSystemPrompt(context.messages)).toContain("Never operate, navigate, alter, or type into the Gloomberb UI");
        return fauxAssistantMessage(fauxToolCall("gloomberb_market_data", {
          operation: "quote",
          symbol: "nvda",
          exchange: "nasdaq",
        }), { stopReason: "toolUse" });
      },
      fauxAssistantMessage(fauxToolCall("submit_screener_results", {
        title: "Semiconductors",
        tickers: [{
          symbol: "NVDA",
          exchange: "NASDAQ",
          reason: "Validated with the configured market data source.",
        }],
      }), { stopReason: "toolUse" }),
    ]);

    await fixture.host.run({
      providerId: "anthropic",
      prompt: "Find a semiconductor",
      outputMode: "screener",
    }).done;

    expect(requests).toEqual([{
      type: "data",
      operation: "quote",
      symbol: "nvda",
      exchange: "nasdaq",
    }]);
  });
});

describe("Pi AI host catalog and account connection", () => {
  test("publishes all seven providers with canonical ids and native capabilities", () => {
    const summaries = AI_PROVIDER_IDS.map((id) => disconnectedSummary(id, id));
    const catalog = toAiRuntimeCatalog({ providers: summaries, refreshErrors: {} });

    expect(catalog.providers.map((provider) => provider.providerId)).toEqual([...AI_PROVIDER_IDS]);
    expect(catalog.providers.every((provider) => (
      provider.outputModes.join(",") === "plain,structured,screener"
    ))).toBe(true);
    expect(catalog.accounts.map((account) => account.providerId)).toEqual([...AI_PROVIDER_IDS]);
    expect(catalog.providers).not.toContainEqual(expect.objectContaining({ providerId: "claude" }));
  });

  test("preserves external API-key readiness without offering disconnect or browser sign-in", async () => {
    const providerSummary: PiProviderSummary = {
      id: "google",
      label: "Google Gemini",
      name: "Google",
      defaultModelId: "gemini-3.6-flash",
      authMethods: [{ type: "api_key", label: "Gemini API key", canLogin: true }],
      connection: {
        state: "connected",
        type: "api_key",
        source: "GEMINI_API_KEY",
        origin: "external",
        disconnectable: false,
      },
      models: [],
    };
    let loginCalls = 0;
    let logoutCalls = 0;
    const runtime = {
      getProviderSummary: async () => providerSummary,
      getCatalog: async () => ({ providers: [providerSummary], refreshErrors: {} }),
      login: async () => { loginCalls += 1; },
      logout: async () => { logoutCalls += 1; },
    };
    const host = createPiAiHost({
      appKind: "tui",
      dataDir: "/tmp/gloomberb-pi-host-external-key-test",
      runtime: runtime as never,
    });

    expect((await host.getCatalog!()).accounts[0]).toMatchObject({
      providerId: "google",
      connectionState: "connected",
      connectionLabel: "Connected with GEMINI_API_KEY",
      credentialOrigin: "external",
      authMethods: [{ type: "api_key", canLogin: true }],
      canLogin: false,
      canSetApiKey: true,
      canDisconnect: false,
    });
    await expect(host.connect!("google")).rejects.toThrow("Enter an API key in AI pane settings instead");
    await expect(host.disconnect!("google")).rejects.toThrow("managed outside Gloomberb");
    expect(loginCalls).toBe(0);
    expect(logoutCalls).toBe(0);

    const storedKeyCatalog = toAiRuntimeCatalog({
      providers: [{
        ...providerSummary,
        connection: {
          state: "connected",
          type: "api_key",
          source: "stored credential",
          origin: "stored",
          disconnectable: true,
        },
      }],
      refreshErrors: {},
    });
    expect(storedKeyCatalog.accounts[0]).toMatchObject({
      credentialOrigin: "stored",
      canDisconnect: true,
    });
  });

  test("saves a typed API key through the runtime and returns only catalog status", async () => {
    const saved: Array<[string, string]> = [];
    const connectedSummary: PiProviderSummary = {
      id: "anthropic",
      label: "Claude",
      name: "Anthropic",
      defaultModelId: "claude-sonnet-5-5",
      authMethods: [
        { type: "oauth", label: "Anthropic (Claude Pro/Max)", canLogin: true },
        { type: "api_key", label: "Anthropic API key", canLogin: true },
      ],
      connection: {
        state: "connected",
        type: "api_key",
        source: "stored credential",
        origin: "stored",
        disconnectable: true,
      },
      models: [],
    };
    const runtime = {
      getProviderSummary: async () => connectedSummary,
      getCatalog: async () => ({ providers: [connectedSummary], refreshErrors: {} }),
      saveApiKey: async (providerId: string, apiKey: string) => {
        saved.push([providerId, apiKey]);
        return connectedSummary;
      },
    };
    const host = createPiAiHost({
      appKind: "tui",
      dataDir: "/tmp/gloomberb-pi-host-save-key-test",
      runtime: runtime as never,
    });

    const catalog = await host.saveApiKey!("anthropic", "sk-ant-api03-secret");

    expect(saved).toEqual([["anthropic", "sk-ant-api03-secret"]]);
    expect(catalog.accounts[0]).toMatchObject({
      providerId: "anthropic",
      connectionLabel: "Connected with an API key saved in Gloomberb",
      canLogin: true,
      canSetApiKey: true,
      canDisconnect: true,
    });
    expect(JSON.stringify(catalog)).not.toContain("sk-ant-api03");
  });

  test("refuses to save a key while an AI Agent reply could drive the app", async () => {
    const saved: string[] = [];
    let finishRun: () => void = () => {};
    const connected: PiProviderSummary = {
      ...disconnectedSummary("anthropic", "Claude"),
      connection: { state: "connected", type: "oauth", origin: "stored", disconnectable: true },
    };
    const runtime = {
      getProviderSummary: async () => connected,
      getCatalog: async () => ({ providers: [connected], refreshErrors: {} }),
      runAgent: () => ({
        done: new Promise((resolve) => {
          finishRun = () => resolve({ text: "done", messages: [] });
        }),
        cancel() {},
      }),
      saveApiKey: async (providerId: string) => {
        saved.push(providerId);
        return connected;
      },
    };
    const host = createPiAiHost({
      appKind: "tui",
      dataDir: "/tmp/gloomberb-pi-host-agent-guard-test",
      runtime: runtime as never,
    });

    const run = host.run({ providerId: "anthropic", prompt: "Drive the app", outputMode: "structured" });
    // The run starts after the provider check settles.
    await new Promise((resolve) => setTimeout(resolve, 0));
    await expect(host.saveApiKey!("anthropic", "sk-ant-attacker"))
      .rejects.toThrow("Wait for the AI Agent reply to finish or cancel it");
    expect(saved).toEqual([]);

    finishRun();
    await run.done;
    await host.saveApiKey!("anthropic", "sk-ant-user");
    expect(saved).toEqual(["anthropic"]);
  });

  test("refuses to save a key while that provider's browser sign-in is pending", async () => {
    const providerSummary = disconnectedSummary("xai", "xAI / Grok");
    const saved: string[] = [];
    const runtime = {
      getProviderSummary: async () => providerSummary,
      getCatalog: async () => ({ providers: [providerSummary], refreshErrors: {} }),
      login: () => new Promise(() => {}),
      saveApiKey: async (providerId: string) => {
        saved.push(providerId);
        return providerSummary;
      },
    };
    const host = createPiAiHost({
      appKind: "tui",
      dataDir: "/tmp/gloomberb-pi-host-pending-login-test",
      openExternal: async () => {},
      runtime: runtime as never,
    });

    void host.connect!("xai");
    await new Promise((resolve) => setTimeout(resolve, 0));

    await expect(host.saveApiKey!("xai", "xai-key"))
      .rejects.toThrow("browser sign-in is still in progress");
    expect(saved).toEqual([]);
  });

  test("reports which kind of credential is connected", () => {
    const catalog = toAiRuntimeCatalog({
      providers: [{
        ...disconnectedSummary("anthropic", "Claude"),
        connection: { state: "connected", type: "oauth", origin: "stored", disconnectable: true },
      }],
      refreshErrors: {},
    });

    expect(catalog.accounts[0]).toMatchObject({ credentialType: "oauth" });
  });

  test("keeps a disconnected provider visible and fails runs clearly without a CLI fallback", async () => {
    const providerSummary = disconnectedSummary("anthropic", "Claude");
    const runtime = {
      getProviderSummary: async () => providerSummary,
      getCatalog: async () => ({ providers: [providerSummary], refreshErrors: {} }),
    };
    const host = createPiAiHost({
      appKind: "tui",
      dataDir: "/tmp/gloomberb-pi-host-disconnected-test",
      runtime: runtime as never,
    });

    expect((await host.getCatalog!()).providers[0]).toMatchObject({
      providerId: "anthropic",
      status: "not_authenticated",
    });
    await expect(host.run({
      providerId: "anthropic",
      prompt: "Do not use a fallback",
    }).done).rejects.toThrow("Claude is not connected");
  });

  test("uses the safe github.com default for Copilot before opening its device flow", async () => {
    const providerSummary = disconnectedSummary("github-copilot", "GitHub Copilot");
    const opened: string[] = [];
    const promptAnswers: string[] = [];
    const authEvents: unknown[] = [];
    const runtime = {
      getProviderSummary: async () => providerSummary,
      login: async (_request: unknown, interaction: {
        notify(event: unknown): void;
        prompt(prompt: unknown, signal?: AbortSignal): Promise<string>;
      }) => {
        promptAnswers.push(await interaction.prompt({
          type: "text",
          message: "GitHub Enterprise URL/domain (blank for github.com)",
          placeholder: "company.ghe.com",
        }));
        interaction.notify({
          type: "device_code",
          userCode: "ABCD-EFGH",
          verificationUri: "https://github.com/login/device",
          intervalSeconds: 5,
          expiresInSeconds: 900,
        });
      },
      getCatalog: async () => ({ providers: [providerSummary], refreshErrors: {} }),
    };
    const host = createPiAiHost({
      appKind: "tui",
      dataDir: "/tmp/gloomberb-pi-host-copilot-test",
      runtime: runtime as never,
      openExternal: async (url) => { opened.push(url); },
    });

    await host.connect!("github-copilot", undefined, (event) => {
      authEvents.push(event);
    });

    expect(promptAnswers).toEqual([""]);
    expect(opened).toEqual(["https://github.com/login/device"]);
    expect(authEvents).toEqual([{
      type: "device_code",
      userCode: "ABCD-EFGH",
      verificationUri: "https://github.com/login/device",
      intervalSeconds: 5,
      expiresInSeconds: 900,
    }]);
  });

  test("makes a device-code browser launch failure terminal instead of polling silently", async () => {
    const providerSummary = disconnectedSummary("github-copilot", "GitHub Copilot");
    const runtime = {
      getProviderSummary: async () => providerSummary,
      login: async (_request: unknown, interaction: {
        notify(event: unknown): void;
      }) => {
        interaction.notify({
          type: "device_code",
          userCode: "ABCD-EFGH",
          verificationUri: "https://github.com/login/device",
        });
        await new Promise(() => {});
      },
    };
    const host = createPiAiHost({
      appKind: "tui",
      dataDir: "/tmp/gloomberb-pi-host-device-launch-test",
      runtime: runtime as never,
      openExternal: async () => { throw new Error("Browser could not be opened."); },
    });

    await expect(host.connect!("github-copilot"))
      .rejects.toThrow("Browser could not be opened");
  });

  test("forwards xAI device codes while opening the verification page", async () => {
    const providerSummary = disconnectedSummary("xai", "xAI / Grok");
    const opened: string[] = [];
    const authEvents: unknown[] = [];
    const runtime = {
      getProviderSummary: async () => providerSummary,
      login: async (_request: unknown, interaction: {
        notify(event: unknown): void;
      }) => {
        interaction.notify({
          type: "device_code",
          userCode: "XAI-1234",
          verificationUri: "https://accounts.x.ai/activate",
        });
      },
      getCatalog: async () => ({ providers: [providerSummary], refreshErrors: {} }),
    };
    const host = createPiAiHost({
      appKind: "tui",
      dataDir: "/tmp/gloomberb-pi-host-xai-test",
      runtime: runtime as never,
      openExternal: async (url) => { opened.push(url); },
    });

    await host.connect!("xai", undefined, (event) => {
      authEvents.push(event);
    });

    expect(opened).toEqual(["https://accounts.x.ai/activate"]);
    expect(authEvents).toEqual([{
      type: "device_code",
      userCode: "XAI-1234",
      verificationUri: "https://accounts.x.ai/activate",
    }]);
  });

  test("reports a browser-launch failure without waiting for the manual-code timeout", async () => {
    const providerSummary = disconnectedSummary("anthropic", "Claude");
    const runtime = {
      getProviderSummary: async () => providerSummary,
      login: async (_request: unknown, interaction: {
        notify(event: unknown): void;
        prompt(prompt: unknown, signal?: AbortSignal): Promise<string>;
      }) => {
        interaction.notify({ type: "auth_url", url: "https://example.test/login" });
        await interaction.prompt({ type: "manual_code", message: "Complete sign-in" });
      },
      getCatalog: async () => ({ providers: [], refreshErrors: {} }),
    };
    const host = createPiAiHost({
      appKind: "tui",
      dataDir: "/tmp/gloomberb-pi-host-login-test",
      runtime: runtime as never,
      openExternal: async () => { throw new Error("Browser could not be opened."); },
    });

    await expect(host.connect!("anthropic")).rejects.toThrow("Browser could not be opened");
  });
});

describe("Pi AI host conversation history", () => {
  test("passes prior user and assistant messages structurally before the current prompt", async () => {
    const fixture = createHostFixture();
    fixture.faux.setResponses([
      (context) => {
        expect(context.messages.map((message) => message.role)).toEqual([
          "user",
          "assistant",
          "user",
        ]);
        expect(context.messages[0]).toMatchObject({ role: "user", content: "Earlier question" });
        expect(context.messages[1]).toMatchObject({
          role: "assistant",
          content: [{ type: "text", text: "Earlier answer" }],
        });
        expect(context.messages[2]).toMatchObject({ role: "user", content: "Current question" });
        return fauxAssistantMessage("Current answer");
      },
    ]);

    const output = await fixture.host.run({
      providerId: "anthropic",
      prompt: "Current question",
      messages: [
        { role: "user", content: "Earlier question" },
        { role: "assistant", content: "Earlier answer" },
      ],
    }).done;

    expect(output).toBe("Current answer");
  });

  test("returns and replays native tool-call and tool-result history", async () => {
    const fixture = createHostFixture(async () => ({
      ok: true,
      data: { paneId: "pane-1" },
    }));
    fixture.faux.setResponses([
      fauxAssistantMessage(fauxToolCall("gloomberb_remote", {
        request: {
          type: "call",
          operation: "pane.open",
          input: { paneType: "watchlist" },
        },
      }), { stopReason: "toolUse" }),
      fauxAssistantMessage("Opened the pane."),
    ]);
    let agentMessages: AiAgentHistoryMessage[] = [];

    await fixture.host.run({
      providerId: "anthropic",
      prompt: "Open a watchlist pane",
      outputMode: "structured",
      onAgentMessages: (messages) => {
        agentMessages = messages;
      },
    }).done;

    expect(agentMessages.map((message) => message.role)).toEqual([
      "user",
      "assistant",
      "toolResult",
      "assistant",
    ]);
    expect(agentMessages[1]).toMatchObject({
      role: "assistant",
      content: [expect.objectContaining({
        type: "toolCall",
        name: "gloomberb_remote",
      })],
    });
    expect(agentMessages[2]).toMatchObject({
      role: "toolResult",
      toolName: "gloomberb_remote",
      content: [{ type: "text", text: '{"ok":true,"data":{"paneId":"pane-1"}}' }],
    });

    fixture.faux.setResponses([
      (context) => {
        // Pi 1.x carries the system prompt and tools as a leading system message.
        expect(context.messages.map((message) => message.role)).toEqual([
          "system",
          "user",
          "assistant",
          "toolResult",
          "assistant",
          "user",
        ]);
        expect(context.messages[3]).toMatchObject({
          role: "toolResult",
          toolName: "gloomberb_remote",
        });
        return fauxAssistantMessage("Closed the same pane.");
      },
    ]);

    const output = await fixture.host.run({
      providerId: "anthropic",
      prompt: "Close the pane you just opened",
      outputMode: "structured",
      agentMessages,
    }).done;

    expect(output).toBe("Closed the same pane.");
  });
});

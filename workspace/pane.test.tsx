import { afterEach, describe, expect, test } from "bun:test";
import { act, useState } from "react";
import { PaneFooterBar, PaneFooterProvider } from "gloomberb/test-support";
import { testRender } from "gloomberb/test-support";
import { createInitialState } from "gloomberb/test-support";
import { createStatefulTestPluginRuntime } from "gloomberb/test-support";
import { createDefaultConfig, createPaneInstance } from "gloomberb/types/config";
import { Box } from "gloomberb/ui";
import type { PluginRuntimeAccess } from "gloomberb/react";
import { setAiRunHost, setAiRuntimeCatalog, type AiRunHost } from "../runner";
import {
  LOCAL_AGENT_WORKSPACE_SCHEMA_VERSION,
  LOCAL_AGENT_WORKSPACE_STATE_KEY,
  LocalAgentWorkspacePane,
} from "./pane";
import {
  createLocalAgentThread,
  EMPTY_LOCAL_AGENT_WORKSPACE,
  type LocalAgentWorkspaceState,
} from "./model";
import { TestPaneProvider, createTestTicker } from "gloomberb/test-support";

const PANE_ID = "local-agent-workspace:test";

let testSetup: Awaited<ReturnType<typeof testRender>> | undefined;

function AgentPaneHarness({
  existingWorkspace,
  newThreadId,
  onOpenSettings,
  configureState,
  width = 100,
  height = 16,
}: {
  configureState?: (state: ReturnType<typeof createInitialState>) => void;
  width?: number;
  height?: number;
  existingWorkspace?: LocalAgentWorkspaceState;
  newThreadId?: string;
  onOpenSettings: PluginRuntimeAccess["openPaneSettings"];
}) {
  const [state] = useState(() => {
    const config = createDefaultConfig("/tmp/gloomberb-agent-pane");
    config.layout.instances.push(createPaneInstance("local-agent-workspace", {
      instanceId: PANE_ID,
      title: "AI Agent",
      ...(newThreadId ? { params: { newThreadId } } : {}),
    }));
    const initial = createInitialState(config);
    initial.focusedPaneId = PANE_ID;
    configureState?.(initial);
    return initial;
  });
  const [runtime] = useState(() => {
    const nextRuntime = createStatefulTestPluginRuntime({
      openPaneSettings: onOpenSettings,
    });
    if (existingWorkspace) {
      nextRuntime.setResumeState(
        "ai",
        LOCAL_AGENT_WORKSPACE_STATE_KEY,
        existingWorkspace,
        LOCAL_AGENT_WORKSPACE_SCHEMA_VERSION,
      );
    }
    return nextRuntime;
  });

  return (
    <Box flexDirection="column" width={width} height={height}>
      <TestPaneProvider state={state} paneId={PANE_ID} pluginId="ai" runtime={runtime}>
        <PaneFooterProvider>
          {(footer) => (
            <>
              <LocalAgentWorkspacePane
                paneId={PANE_ID}
                paneType="local-agent-workspace"
                focused
                width={width}
                height={height - 1}
              />
              <PaneFooterBar footer={footer} focused width={width} />
            </>
          )}
        </PaneFooterProvider>
      </TestPaneProvider>
    </Box>
  );
}

afterEach(async () => {
  if (testSetup) {
    await act(async () => {
      testSetup?.renderer.destroy();
    });
    testSetup = undefined;
  }
  setAiRunHost(null);
  setAiRuntimeCatalog({ providers: [], accounts: [], models: [] });
});

describe("LocalAgentWorkspacePane provider setup", () => {
  test("shows disconnected providers, prefers a ready account, and opens shared pane settings", async () => {
    setAiRuntimeCatalog({
      providers: [
        {
          providerId: "anthropic",
          label: "Claude",
          status: "not_authenticated",
          unavailableReason: "Claude is not connected.",
          outputModes: ["plain", "structured", "screener"],
        },
        {
          providerId: "openai-codex",
          label: "OpenAI (ChatGPT)",
          status: "ready",
          outputModes: ["plain", "structured", "screener"],
        },
      ],
      accounts: [],
      models: [],
    });

    const openedSettings: string[] = [];
    testSetup = await testRender(
      <AgentPaneHarness
        existingWorkspace={createLocalAgentThread(
          EMPTY_LOCAL_AGENT_WORKSPACE,
          "openai-codex",
          { id: "existing-thread", now: 1 },
        )}
        newThreadId="new-pane-thread"
        onOpenSettings={(paneId) => {
          if (paneId) openedSettings.push(paneId);
        }}
      />,
      { width: 100, height: 16 },
    );
    await act(async () => {
      await testSetup!.renderOnce();
      await testSetup!.renderOnce();
    });

    const readyFrame = testSetup.captureCharFrame();
    expect(readyFrame).toContain("OpenAI (ChatGPT)");
    expect(readyFrame).toContain("OpenAI (ChatGPT) is ready.");
    const providerRow = readyFrame.split("\n").findIndex((line) => (
      line.trim() === "OpenAI (ChatGPT)"
    ));
    const providerColumn = readyFrame.split("\n")[providerRow]?.indexOf("OpenAI (ChatGPT)") ?? -1;
    expect(providerRow).toBeGreaterThanOrEqual(0);
    expect(providerColumn).toBeGreaterThanOrEqual(0);

    await act(async () => {
      await testSetup!.mockMouse.click(providerColumn + 1, providerRow);
      await testSetup!.renderOnce();
      await testSetup!.renderOnce();
    });
    const providerPickerFrame = testSetup.captureCharFrame();
    expect(providerPickerFrame).toContain("Choose AI provider");
    expect(providerPickerFrame).toContain("Claude · connect");

    await act(async () => {
      testSetup?.renderer.destroy();
    });
    testSetup = undefined;
    setAiRuntimeCatalog({
      providers: [{
        providerId: "anthropic",
        label: "Claude",
        status: "not_authenticated",
        unavailableReason: "Claude is not connected.",
        outputModes: ["plain", "structured", "screener"],
      }],
      accounts: [],
      models: [],
    });

    testSetup = await testRender(
      <AgentPaneHarness onOpenSettings={(paneId) => {
        if (paneId) openedSettings.push(paneId);
      }} />,
      { width: 100, height: 16 },
    );
    await act(async () => {
      await testSetup!.renderOnce();
      await testSetup!.renderOnce();
    });

    const disconnectedFrame = testSetup.captureCharFrame();
    expect(disconnectedFrame).toContain("Claude is not connected.");
    expect(disconnectedFrame).toContain("Configure Claude");

    await act(async () => {
      testSetup!.mockInput.pressKey("s");
      await testSetup!.renderOnce();
    });

    expect(openedSettings).toEqual([PANE_ID]);
  });
});


test("a long research attachment keeps its controls and composer outside the preview", async () => {
  setAiRuntimeCatalog({ providers: [{ providerId: "anthropic", label: "Claude", status: "ready", outputModes: ["plain", "structured", "screener"] }], accounts: [], models: [] });
  let request: Parameters<AiRunHost["run"]>[0] | undefined;
  setAiRunHost({
    async checkStatus() { return { available: true, authenticated: true, message: null }; },
    run(options) { request = options; return { done: Promise.resolve("Controlled local response"), cancel() {} }; },
  });
  testSetup = await testRender(<AgentPaneHarness
    width={48}
    height={14}
    existingWorkspace={createLocalAgentThread(EMPTY_LOCAL_AGENT_WORKSPACE, "anthropic", { id: "attachment", now: 1 })}
    onOpenSettings={() => {}}
    configureState={(state) => {
      state.recentTickers = ["CONTROL"];
      state.tickers.set("CONTROL", createTestTicker("CONTROL", "Controlled issuer"));
      state.financials.set("CONTROL", {
        fundamentals: { financialCurrency: "JPY", eps: 42.5, netIncome: 0, stale: true },
        annualStatements: [{ date: "2025-12-31", currency: "JPY", totalRevenue: 100, netIncome: 0, totalDebt: 0 }],
        quarterlyStatements: [], priceHistory: [],
      });
    }}
  />, { width: 48, height: 14, screenMode: "alternate-screen" });
  await act(async () => { await testSetup!.renderOnce(); });
  const clickText = async (text: string) => {
    const lines = testSetup!.captureCharFrame().split("\n");
    const row = lines.findIndex((line) => line.includes(text));
    expect(row).toBeGreaterThanOrEqual(0);
    await act(async () => { await testSetup!.mockMouse.click(lines[row]!.indexOf(text) + 1, row); await testSetup!.renderOnce(); });
    await testSetup!.renderOnce();
  };
  // Attach is the footer's [a] action, not a button row above the composer.
  await clickText("[a]ttach CONTROL");
  const attached = testSetup.captureCharFrame();
  expect(attached).toContain("Attached: Ticker CONTROL");
  expect(attached).toContain("Company: Controlled issuer");
  expect(attached).toContain("Message Claude");
  await clickText("Remove");
  expect(testSetup.captureCharFrame()).not.toContain("Attached: Ticker CONTROL");
  await clickText("[a]ttach CONTROL");
  await act(async () => { testSetup!.mockInput.pressEnter(); await testSetup!.renderOnce(); });
  await act(async () => { await testSetup!.mockInput.typeText("Compare earnings"); testSetup!.mockInput.pressEnter(); await testSetup!.renderOnce(); });
  expect(request?.prompt).toContain("EPS: 42.5 JPY");
  expect(request?.prompt).toContain("Net Income: 0 JPY");
});

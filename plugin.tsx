import type { AppConfig } from "gloomberb/types/config";
import type { CliCommandDef, GloomPlugin, GloomPluginContext } from "gloomberb/types/plugin";
import { getPluginPaneStateValue } from "gloomberb/react";
import {
  LIVE_STREAMING_QUICK_SETTING,
  withLiveStreamingSetting,
} from "gloomberb/quotes";
import { AskAiResearchTab } from "./ask-ai-detail-tab";
import {
  AI_DEFAULT_MODEL_SETTING_KEY,
  AI_DEFAULT_PROVIDER_SETTING_KEY,
  AI_PANE_MODEL_SETTING_KEY,
  AI_PANE_PROVIDER_SETTING_KEY,
  buildAiPaneSettingsDef,
  resolveAiSharedDefaults,
  type AiAccountSettingRow,
  type AiSharedDefaults,
} from "./pane-settings";
import {
  detectProviders,
  getAiProviderDefinitions,
  resolveDefaultAiProviderId,
  type AiProvider,
  type AiProviderId,
} from "./providers";
import {
  connectAiRuntimeProvider,
  disconnectAiRuntimeProvider,
  ensureAiRunHost,
  getAiRuntimeCatalog,
  getAiRuntimeCatalogSnapshot,
  saveAiRuntimeApiKey,
  subscribeAiRuntimeCatalog,
  type AiRuntimeAccount,
} from "./runner";
import {
  buildAiRunnerWizard,
  getSelectableAiRunners,
  resolveAiRunnerWizardModel,
  resolveReadyAiRunnerDefault,
  supportsAiRunOutputMode,
} from "./runner-selection";
import {
  EMPTY_PANE_STATE,
  normalizeTabs,
  type PersistedAiScreenerPaneState,
} from "./screener/model";
import {
  AI_SCREENER_PANE_STATE_KEY,
  AiScreenerPane,
} from "./screener/pane";
import { buildAiScreenerPaneSettingsDef, getAiScreenerPaneSettings } from "./settings";
import {
  EMPTY_LOCAL_AGENT_WORKSPACE,
  normalizeLocalAgentWorkspace,
  type LocalAgentWorkspaceState,
} from "./workspace/model";
import {
  LOCAL_AGENT_WORKSPACE_SCHEMA_VERSION,
  LOCAL_AGENT_WORKSPACE_STATE_KEY,
  LocalAgentWorkspacePane,
} from "./workspace/pane";

function settingOrFallback(
  settings: Record<string, unknown>,
  key: string,
  fallback: string | null | undefined,
): string {
  if (!Object.prototype.hasOwnProperty.call(settings, key)) return fallback?.trim() ?? "";
  return typeof settings[key] === "string" ? settings[key].trim() : "";
}

function modelSettingOrFallback(
  settings: Record<string, unknown>,
  savedProviderId: string | null | undefined,
  savedModelId: string | null | undefined,
): string {
  if (Object.prototype.hasOwnProperty.call(settings, AI_PANE_MODEL_SETTING_KEY)) {
    return settingOrFallback(settings, AI_PANE_MODEL_SETTING_KEY, null);
  }
  if (Object.prototype.hasOwnProperty.call(settings, AI_PANE_PROVIDER_SETTING_KEY)) {
    const configuredProviderId = settingOrFallback(settings, AI_PANE_PROVIDER_SETTING_KEY, null);
    if (configuredProviderId !== (savedProviderId?.trim() ?? "")) return "";
  }
  return savedModelId?.trim() ?? "";
}

function settingsProviders(providers: readonly AiProvider[]) {
  const accounts = new Map(
    getAiRuntimeCatalog().accounts.map((account) => [account.providerId, account] as const),
  );
  return providers.map((provider) => ({
    id: provider.id,
    label: provider.name,
    description: accounts.get(provider.id)?.connectionState === "connected"
      ? accounts.get(provider.id)?.connectionLabel
      : provider.status === "ready"
        ? `${provider.name} is ready.`
        : accounts.get(provider.id)?.canLogin
          ? `${provider.name} can be connected here.`
          : accounts.get(provider.id)?.canSetApiKey
            ? `${provider.name} can be connected here with an API key.`
          : `${provider.name} is not currently available.`,
  }));
}

function settingsModels(providerIds: ReadonlySet<string>) {
  return getAiRuntimeCatalog().models
    .filter((model) => providerIds.has(model.providerId))
    .map((model) => ({
      id: model.id,
      providerId: model.providerId,
      label: model.label,
      description: model.available ? "Available for the connected account." : undefined,
    }));
}

function signInSettingRow(account: AiRuntimeAccount): AiAccountSettingRow {
  const disconnectable = account.canDisconnect;
  const oauthLabel = account.authMethods.find((method) => method.type === "oauth")?.label;
  return {
    providerId: account.providerId,
    providerLabel: account.providerLabel,
    // Next to an API-key row, say which account a browser sign-in uses.
    description: account.connectionState === "not_connected" && account.canLogin && oauthLabel
      ? `Browser sign-in: ${oauthLabel}.`
      : account.connectionLabel,
    actionLabel: disconnectable ? "Disconnect" : "Connect",
    async action(context) {
      context.close();
      if (disconnectable) {
        context.notify({
          body: `Disconnecting ${account.providerLabel}…`,
          type: "info",
          duration: 5_000,
        });
        try {
          const catalog = await disconnectAiRuntimeProvider(account.providerId);
          const remainingAccount = catalog.accounts.find((candidate) => (
            candidate.providerId === account.providerId
          ));
          context.notify({
            body: remainingAccount?.connectionState === "connected"
              ? `${account.providerLabel}'s Gloomberb account was removed. ${remainingAccount.connectionLabel} remains active.`
              : `${account.providerLabel} is disconnected from Gloomberb.`,
            type: "success",
          });
        } catch (error) {
          context.notify({
            body: error instanceof Error ? error.message : `${account.providerLabel} disconnection failed.`,
            type: "error",
            persistent: true,
          });
        }
        return;
      }
      context.notify({
        body: `Opening ${account.providerLabel} sign-in in your browser…`,
        type: "info",
        duration: 5_000,
      });
      try {
        await connectAiRuntimeProvider(account.providerId, undefined, (event) => {
          if (event.type !== "device_code") return;
          context.notify({
            title: `${account.providerLabel} device sign-in`,
            body: `Enter code ${event.userCode} at ${event.verificationUri}. The sign-in page has been opened in your browser.`,
            type: "info",
            persistent: true,
          });
        });
        context.notify({
          body: `${account.providerLabel} is connected.`,
          type: "success",
        });
      } catch (error) {
        context.notify({
          body: error instanceof Error ? error.message : `${account.providerLabel} sign-in failed.`,
          type: "error",
          persistent: true,
        });
      }
    },
  };
}

function accountSettingRows(providerIds: ReadonlySet<string>): AiAccountSettingRow[] {
  return getAiRuntimeCatalog().accounts
    .filter((account) => providerIds.has(account.providerId))
    .flatMap((account) => [
      ...(account.canLogin || account.canDisconnect ? [signInSettingRow(account)] : []),
      ...(account.canSetApiKey ? [apiKeySettingRow(account)] : []),
    ]);
}

/**
 * What each provider calls its key and where it is issued. Static because the
 * commands are registered before the runtime catalog has loaded. ChatGPT is
 * absent: Pi signs it in through the browser only.
 */
const API_KEY_COPY: Readonly<Partial<Record<AiProviderId, { noun: string; hint: string }>>> = {
  anthropic: {
    noun: "Claude API key",
    hint: "Create one in the Claude Console. Usage is billed to that API account, not to a Claude subscription.",
  },
  openai: { noun: "OpenAI API key", hint: "Create one in the OpenAI Platform dashboard." },
  google: { noun: "Gemini API key", hint: "Create one in Google AI Studio." },
  "github-copilot": {
    noun: "GitHub Copilot token",
    hint: "Use a GitHub token with Copilot access, the one Pi reads from COPILOT_GITHUB_TOKEN.",
  },
  xai: { noun: "xAI API key", hint: "Create one in the xAI Console." },
  openrouter: { noun: "OpenRouter API key", hint: "Create one on OpenRouter's Keys page." },
};

function apiKeyCopy(account: AiRuntimeAccount): { noun: string; hint: string } {
  return API_KEY_COPY[account.providerId] ?? {
    noun: `${account.providerLabel} API key`,
    hint: `Paste an API key for ${account.providerLabel}.`,
  };
}

function apiKeyCommandLabel(noun: string): string {
  return `Set ${noun}`;
}

function hasSavedApiKey(account: AiRuntimeAccount): boolean {
  return account.connectionState === "connected"
    && account.credentialType === "api_key"
    && account.credentialOrigin === "stored";
}

function apiKeySettingRow(account: AiRuntimeAccount): AiAccountSettingRow {
  const saved = hasSavedApiKey(account);
  const copy = apiKeyCopy(account);
  return {
    providerId: account.providerId,
    providerLabel: account.providerLabel,
    variant: "api-key",
    label: copy.noun,
    description: saved
      ? account.connectionLabel
      : account.connectionState === "connected"
        // A saved key wins over a sign-in or an environment variable.
        ? `Takes precedence over the current connection (${account.connectionLabel}).`
        : copy.hint,
    actionLabel: saved ? "Replace key" : "Enter key",
    action(context) {
      // Pane settings have no masked field, so the key goes into the command
      // bar's password step instead. The query lands on that command.
      context.close();
      context.openCommandBar(apiKeyCommandLabel(copy.noun));
    },
  };
}

/**
 * One masked-entry command per provider, reachable from its settings row or
 * typed into the command bar, and hidden where Pi takes no key.
 */
function registerApiKeyCommands(ctx: GloomPluginContext): void {
  for (const definition of getAiProviderDefinitions()) {
    const copy = API_KEY_COPY[definition.id];
    if (!copy) continue;
    ctx.registerCommand({
      id: `ai-api-key:${definition.id}`,
      label: apiKeyCommandLabel(copy.noun),
      keywords: ["ai", "api key", "key", "token", "connect", definition.id, definition.name.toLowerCase()],
      category: "config",
      description: `Connect ${definition.name} with an API key instead of a browser sign-in.`,
      hidden: () => !getAiRuntimeCatalogSnapshot().accounts.some((account) => (
        account.providerId === definition.id && account.canSetApiKey
      )),
      wizard: [{
        key: "apiKey",
        label: copy.noun,
        type: "password",
        placeholder: "Paste the key",
        // The command bar shows only the first body line of an input step.
        body: [`${copy.hint} Saved in Gloomberb's data directory, replacing any credential saved there for this provider.`],
      }],
      async execute(values) {
        await ensureAiRunHost();
        // Errors surface in the wizard; Pi's messages never contain the key.
        await saveAiRuntimeApiKey(definition.id, values?.apiKey ?? "");
        // Pi stores the key without calling the provider, so a typo only
        // shows up when a request is made.
        ctx.notify({
          body: `${copy.noun} saved. It is checked on the first request.`,
          type: "success",
        });
      },
    });
  }
}

function defaultsFromConfig(
  config: AppConfig,
  fallbackProviderId: string,
): AiSharedDefaults {
  return resolveAiSharedDefaults(config.pluginConfig.ai, fallbackProviderId);
}

let disposeRuntimeCatalog: (() => void) | undefined;

/**
 * Installs whatever runs prompts in this process: the Pi runtime in the
 * terminal and the Bun half of the desktop app, the RPC bridge to it in the
 * desktop view. Both halves register the same panes, so the difference stays
 * in one argument instead of two copies of the plugin.
 */
export type SetupAiRunHost = (ctx: GloomPluginContext) => void;

export interface ByokAiPluginOptions {
  setupRunHost: SetupAiRunHost;
  /**
   * Only the Bun entry carries these: the command loads the Pi runtime, which
   * the renderer bundle cannot compile and would never run anyway.
   */
  cliCommands?: CliCommandDef[];
}

export function createByokAiPlugin({ setupRunHost, cliCommands }: ByokAiPluginOptions): GloomPlugin {
  return {
    id: "byok-ai",
    name: "BYOK AI",
    version: "1.0.0",
    description: "Bring your own Anthropic, OpenAI, Google, GitHub Copilot, xAI, or OpenRouter account for in-app AI",
    homepage: "https://github.com/gloom-sh/gloom-byok-ai",
    toggleable: true,
    // The panes, threads, screeners, and provider defaults saved while this was
    // a built-in are all under `ai`, and they stay there: only the plugin id
    // moved.
    stateId: "ai",
    // Everything here runs through a provider account on the user's own machine,
    // which the hosted web app has no Bun process to reach.
    targets: ["cli", "tui", "desktop"],
    ...(cliCommands ? { cliCommands } : {}),

    setup(ctx) {
      setupRunHost(ctx);
      const initialProviders = detectProviders();
      const initialScreenerRunners = getSelectableAiRunners(initialProviders, { outputMode: "screener" });
      const fallbackProviderId = resolveDefaultAiProviderId(initialScreenerRunners);
      const initialDefaults = resolveAiSharedDefaults(
        ctx.getConfig().pluginConfig.ai,
        fallbackProviderId,
      );
      const screenerRunnerWizard = buildAiRunnerWizard(initialProviders, {
        defaultProviderId: initialDefaults.providerId,
        defaultModelId: initialDefaults.modelId,
        outputMode: "screener",
      });
      const screenerWizard = [
        ...screenerRunnerWizard,
        {
          key: "prompt",
          label: "Screener Prompt",
          type: "textarea" as const,
          placeholder: "Examples: humanoid robot suppliers, defense software compounders, EM payment rails, obesity-drug picks-and-shovels...",
          body: [
            "Describe the screening idea in plain English.",
            "The AI will return validated ticker ideas with a short reason for each one.",
          ],
        },
      ];
      const updateWizards = (config: AppConfig) => {
        const providers = detectProviders();
        const screenerRunners = getSelectableAiRunners(providers, { outputMode: "screener" });
        const currentFallbackProviderId = resolveDefaultAiProviderId(screenerRunners);
        const defaults = defaultsFromConfig(config, currentFallbackProviderId);
        const screenerDefaults = resolveReadyAiRunnerDefault(
          screenerRunners,
          defaults.providerId,
          defaults.modelId,
        );
        const nextRunnerWizard = buildAiRunnerWizard(providers, {
          defaultProviderId: screenerDefaults.providerId,
          defaultModelId: screenerDefaults.modelId,
          outputMode: "screener",
        });
        screenerWizard.splice(0, Math.max(0, screenerWizard.length - 1), ...nextRunnerWizard);
      };

      const seedDefaultProvider = ctx.configState.get(AI_DEFAULT_PROVIDER_SETTING_KEY) === null;
      const seedDefaultModel = ctx.configState.get(AI_DEFAULT_MODEL_SETTING_KEY) === null;
      if (seedDefaultProvider || seedDefaultModel) {
        void (async () => {
          try {
            if (seedDefaultProvider) {
              await ctx.configState.set(AI_DEFAULT_PROVIDER_SETTING_KEY, initialDefaults.providerId);
            }
            if (seedDefaultModel) {
              await ctx.configState.set(AI_DEFAULT_MODEL_SETTING_KEY, initialDefaults.modelId ?? "");
            }
          } catch (error) {
            ctx.log.warn("Failed to save the default AI provider settings.", error);
          }
        })();
      }
      ctx.on("config:changed", ({ config }) => updateWizards(config));
      disposeRuntimeCatalog?.();
      disposeRuntimeCatalog = subscribeAiRuntimeCatalog(() => updateWizards(ctx.getConfig()));

      registerApiKeyCommands(ctx);

      ctx.registerTickerResearchTab({
        id: "ai-chat",
        name: "Ask AI",
        order: 60,
        component: AskAiResearchTab,
      });

      ctx.registerPane({
        id: "local-agent-workspace",
        name: "AI Agent",
        icon: "A",
        component: LocalAgentWorkspacePane,
        defaultPosition: "right",
        defaultMode: "floating",
        defaultFloatingSize: { width: 96, height: 32 },
        portableShare: {
          private: { title: true, params: true, settings: true, state: true },
        },
        settings: (context) => {
          const providers = detectProviders();
          const workspaceProviders = providers.filter((provider) => (
            supportsAiRunOutputMode(provider, "structured")
          ));
          const workspaceRunners = getSelectableAiRunners(workspaceProviders, {
            outputMode: "structured",
          });
          const workspaceFallbackProviderId = resolveDefaultAiProviderId(workspaceRunners);
          const defaults = defaultsFromConfig(context.config, workspaceFallbackProviderId);
          const workspaceDefaults = workspaceProviders.some((provider) => provider.id === defaults.providerId)
            ? defaults
            : { providerId: workspaceFallbackProviderId, modelId: null };
          const workspaceProviderIds = new Set(workspaceProviders.map((provider) => provider.id));
          const persisted = ctx.resume.getState<LocalAgentWorkspaceState>(
            LOCAL_AGENT_WORKSPACE_STATE_KEY,
            { schemaVersion: LOCAL_AGENT_WORKSPACE_SCHEMA_VERSION },
          ) ?? EMPTY_LOCAL_AGENT_WORKSPACE;
          const workspace = normalizeLocalAgentWorkspace(persisted);
          const activeThreadId = typeof context.paneState.activeThreadId === "string"
            ? context.paneState.activeThreadId
            : workspace.activeThreadId;
          const activeThread = workspace.threads.find((thread) => thread.id === activeThreadId)
            ?? workspace.threads[0]
            ?? null;
          return buildAiPaneSettingsDef({
            title: "AI Agent Settings",
            providers: settingsProviders(workspaceProviders),
            models: settingsModels(workspaceProviderIds),
            defaultProviderId: workspaceDefaults.providerId,
            defaultModelId: workspaceDefaults.modelId,
            paneOverride: {
              providerId: settingOrFallback(context.settings, AI_PANE_PROVIDER_SETTING_KEY, activeThread?.providerId),
              modelId: modelSettingOrFallback(
                context.settings,
                activeThread?.providerId,
                activeThread?.modelId,
              ),
            },
            accountRows: accountSettingRows(workspaceProviderIds),
          });
        },
      });

      ctx.registerPaneTemplate({
        id: "new-local-agent-workspace",
        paneId: "local-agent-workspace",
        label: "AI Agent",
        description: "Create a persistent AI thread with optional model selection.",
        keywords: ["ai", "agent", "claude", "openai", "chatgpt", "gemini", "copilot", "grok", "openrouter", "research", "thread"],
        shortcut: { prefix: "AGENT" },
        createInstance: () => ({
          title: "AI Agent",
          placement: "floating",
          params: { newThreadId: crypto.randomUUID() },
        }),
      });

      ctx.registerPane({
        id: "ai-screener",
        name: "AI Screener",
        icon: "A",
        component: AiScreenerPane,
        defaultPosition: "right",
        defaultMode: "floating",
        defaultFloatingSize: { width: 76, height: 24 },
        tickerSource: true,
        quickSettings: [LIVE_STREAMING_QUICK_SETTING],
        portableShare: {
          private: { title: true, params: true, settings: true, state: true },
        },
        settings: (context) => {
          const providers = detectProviders();
          const screenerProviders = providers.filter((provider) => (
            supportsAiRunOutputMode(provider, "screener")
          ));
          const screenerRunners = getSelectableAiRunners(screenerProviders, {
            outputMode: "screener",
          });
          const currentFallbackProviderId = resolveDefaultAiProviderId(screenerRunners);
          const defaults = defaultsFromConfig(context.config, currentFallbackProviderId);
          const providerIds = new Set(screenerProviders.map((provider) => provider.id));
          const persisted = getPluginPaneStateValue<PersistedAiScreenerPaneState | null>(
            context.paneState,
            "ai",
            AI_SCREENER_PANE_STATE_KEY,
            null,
          ) ?? ctx.resume.getState<PersistedAiScreenerPaneState>(
            `screener-pane:${context.paneId}`,
            { schemaVersion: 1 },
          ) ?? EMPTY_PANE_STATE;
          const tabs = normalizeTabs(persisted);
          const activeTabId = typeof context.paneState.activeTabId === "string"
            ? context.paneState.activeTabId
            : null;
          const activeTab = tabs.find((tab) => tab.id === activeTabId) ?? tabs[0] ?? null;
          return withLiveStreamingSetting(buildAiPaneSettingsDef({
            title: "AI Screener Settings",
            providers: settingsProviders(screenerProviders),
            models: settingsModels(providerIds),
            defaultProviderId: defaults.providerId,
            defaultModelId: defaults.modelId,
            paneOverride: {
              providerId: settingOrFallback(context.settings, AI_PANE_PROVIDER_SETTING_KEY, activeTab?.providerId),
              modelId: modelSettingOrFallback(
                context.settings,
                activeTab?.providerId,
                activeTab?.modelId,
              ),
            },
            accountRows: accountSettingRows(providerIds),
            additional: buildAiScreenerPaneSettingsDef(
              getAiScreenerPaneSettings(context.settings),
            ),
          }), context.settings);
        },
      });

      ctx.registerPaneTemplate({
        id: "new-ai-screener-pane",
        paneId: "ai-screener",
        label: "AI Screener",
        description: "Create a prompt-driven AI screener pane with reusable screening tabs.",
        keywords: ["ai", "screener", "screen", "watchlist", "prompt"],
        shortcut: { prefix: "AI", argPlaceholder: "prompt", argKind: "text" },
        wizard: screenerWizard,
        createInstance: (context, options) => {
          const prompt = options?.values?.prompt?.trim() || options?.arg?.trim() || "";
          if (!prompt) return null;
          const providers = detectProviders();
          const screenerRunners = getSelectableAiRunners(providers, {
            outputMode: "screener",
          });
          const currentFallbackProviderId = resolveDefaultAiProviderId(screenerRunners);
          const defaults = resolveAiSharedDefaults(
            context.config.pluginConfig.ai,
            currentFallbackProviderId,
          );
          const readyDefaults = resolveReadyAiRunnerDefault(
            screenerRunners,
            defaults.providerId,
            defaults.modelId,
          );
          const providerId = options?.values?.providerId || readyDefaults.providerId;
          const modelId = resolveAiRunnerWizardModel(
            options?.values,
            providerId,
            providerId === readyDefaults.providerId ? readyDefaults.modelId : null,
          );
          return {
            title: "AI Screener",
            placement: "floating",
            params: {
              prompt,
              providerId,
              ...(modelId ? { modelId } : {}),
            },
          };
        },
      });
    },
    dispose() {
      disposeRuntimeCatalog?.();
      disposeRuntimeCatalog = undefined;
    },
  };
}

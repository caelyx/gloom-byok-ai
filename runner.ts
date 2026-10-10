import {
  aiProviderFromRuntime,
  isAiProviderId,
  migrateLegacyAiProviderId,
  setDetectedProviders,
  type AiProvider,
  type AiProviderId,
  type AiProviderStatus,
} from "./providers";
import type { AiAgentHistoryMessage } from "./agent-history";
import { withDeadline } from "gloomberb/utils";

export class AiRunCancelledError extends Error {
  constructor() {
    super("AI run cancelled");
    this.name = "AiRunCancelledError";
  }
}

export interface AiRunController {
  done: Promise<string>;
  cancel: () => void;
}

export interface AiConversationMessage {
  role: "user" | "assistant";
  content: string;
}

export type AiRunOutputMode = "plain" | "structured" | "screener";
export type AiRuntimeConnectionState = "connected" | "not_connected" | "error";
export type AiRuntimeAuthType = "oauth" | "api_key";
export type AiCredentialOrigin = "stored" | "external";

export type AiAuthProgressEvent =
  | {
      type: "info";
      message: string;
      links?: readonly { url: string; label?: string }[];
    }
  | {
      type: "auth_url";
      url: string;
      instructions?: string;
    }
  | {
      type: "device_code";
      userCode: string;
      verificationUri: string;
      intervalSeconds?: number;
      expiresInSeconds?: number;
    }
  | {
      type: "progress";
      message: string;
    };

export interface AiRuntimeModel {
  id: string;
  providerId: AiProviderId;
  label: string;
  available: boolean;
}

export interface AiRuntimeAuthMethod {
  type: AiRuntimeAuthType;
  label: string;
  /**
   * True only when the current renderer-neutral host can complete this method
   * without exposing a credential in an unmasked field. API keys qualify
   * because they are typed into the command bar's masked password step.
   */
  canLogin: boolean;
}

export interface AiRuntimeAccount {
  providerId: AiProviderId;
  providerLabel: string;
  connectionState: AiRuntimeConnectionState;
  connectionLabel: string;
  credentialSource?: string;
  credentialOrigin?: AiCredentialOrigin;
  authMethods: AiRuntimeAuthMethod[];
  /** Browser or device-code sign-in is available. */
  canLogin: boolean;
  /** An API key can be typed in and stored for this provider. */
  canSetApiKey: boolean;
  canDisconnect: boolean;
  loginType?: AiRuntimeAuthType;
}

export interface AiRuntimeProvider {
  providerId: AiProviderId;
  label: string;
  status: AiProviderStatus;
  unavailableReason?: string;
  outputModes: AiRunOutputMode[];
  defaultModelId?: string;
}

export interface AiRuntimeCatalog {
  providers: AiRuntimeProvider[];
  accounts: AiRuntimeAccount[];
  models: AiRuntimeModel[];
}

export interface AiRunHost {
  run(options: {
    providerId: AiProviderId;
    prompt: string;
    messages?: AiConversationMessage[];
    agentMessages?: AiAgentHistoryMessage[];
    modelId?: string;
    onChunk?: (output: string) => void;
    onAgentMessages?: (messages: AiAgentHistoryMessage[]) => void;
    outputMode?: AiRunOutputMode;
  }): AiRunController;
  checkStatus?(providerId: AiProviderId): Promise<AiProviderStatusResult>;
  getCatalog?(): Promise<AiRuntimeCatalog>;
  connect?(
    providerId: AiProviderId,
    authType?: AiRuntimeAuthType,
    onAuthEvent?: (event: AiAuthProgressEvent) => void,
  ): Promise<AiRuntimeCatalog>;
  disconnect?(providerId: AiProviderId): Promise<AiRuntimeCatalog>;
  /** Stores a typed API key in the runtime's credential store. */
  saveApiKey?(providerId: AiProviderId, apiKey: string): Promise<AiRuntimeCatalog>;
}

export interface AiProviderStatusResult {
  available: boolean;
  authenticated: boolean;
  /** True when the provider exists but credential validation failed unexpectedly. */
  inconclusive?: boolean;
  message: string | null;
}

type CatalogListener = () => void;

let configuredHost: AiRunHost | null = null;
let runtimeCatalog: AiRuntimeCatalog = { providers: [], accounts: [], models: [] };
const catalogListeners = new Set<CatalogListener>();

function canonicalProviderId(providerId: string): AiProviderId {
  const canonicalId = migrateLegacyAiProviderId(providerId);
  if (!isAiProviderId(canonicalId)) {
    throw new Error(`Unknown AI provider: ${providerId}`);
  }
  return canonicalId;
}

function cloneCatalog(catalog: AiRuntimeCatalog): AiRuntimeCatalog {
  return {
    providers: catalog.providers.map((provider) => ({
      ...provider,
      outputModes: [...provider.outputModes],
    })),
    accounts: catalog.accounts.map((account) => ({
      ...account,
      authMethods: account.authMethods.map((method) => ({ ...method })),
    })),
    models: catalog.models.map((model) => ({ ...model })),
  };
}

function publishCatalog(catalog: AiRuntimeCatalog): void {
  runtimeCatalog = cloneCatalog(catalog);
  setDetectedProviders(runtimeCatalog.providers.map(aiProviderFromRuntime));
  for (const listener of [...catalogListeners]) listener();
}

export function setAiRunHost(host: AiRunHost | null): void {
  configuredHost = host;
}

type AiRunHostLoader = () => Promise<unknown>;

let hostLoader: AiRunHostLoader | null = null;
let hostLoad: Promise<unknown> | null = null;

/**
 * Registers how this process gets its host, without loading it yet.
 *
 * The Pi runtime behind the terminal host is about 830 modules and costs real
 * main-thread time to evaluate, which a plugin must not spend during app
 * startup. The entry points register a loader here and the app calls it once
 * the first layout is up, or sooner if a pane asks for a runtime first.
 */
export function setAiRunHostLoader(loader: AiRunHostLoader | null): void {
  hostLoader = loader;
  hostLoad = null;
}

/** Loads the host at most once. Discovery publishes to the catalog when it lands. */
export function ensureAiRunHost(): Promise<unknown> {
  if (!hostLoader) return Promise.resolve();
  hostLoad ??= hostLoader();
  return hostLoad;
}

export async function installAiRunHost(
  host: AiRunHost,
  options: {
    catalogTimeoutMs: number;
    timeoutMessage: string;
    onCatalogError?: (error: unknown) => void;
  },
): Promise<AiRuntimeCatalog> {
  setAiRunHost(host);
  const emptyCatalog: AiRuntimeCatalog = { providers: [], accounts: [], models: [] };
  const catalog = await withDeadline(
    Promise.resolve().then(() => host.getCatalog?.() ?? emptyCatalog),
    options.catalogTimeoutMs,
    options.timeoutMessage,
  ).catch((error) => {
    try {
      options.onCatalogError?.(error);
    } catch {
      // Logging must not turn optional provider discovery into a startup failure.
    }
    return emptyCatalog;
  });
  setAiRuntimeCatalog(catalog);
  return getAiRuntimeCatalog();
}

export function setAiRuntimeCatalog(catalog: AiRuntimeCatalog): void {
  publishCatalog(catalog);
}

/**
 * Stable snapshot for useSyncExternalStore. A new reference is published only
 * when set/connect/disconnect replaces the catalog.
 */
export function getAiRuntimeCatalogSnapshot(): AiRuntimeCatalog {
  return runtimeCatalog;
}

export function subscribeAiRuntimeCatalog(listener: CatalogListener): () => void {
  catalogListeners.add(listener);
  return () => {
    catalogListeners.delete(listener);
  };
}

/** Defensive copy for imperative consumers. */
export function getAiRuntimeCatalog(): AiRuntimeCatalog {
  return cloneCatalog(runtimeCatalog);
}

export async function connectAiRuntimeProvider(
  providerId: string,
  authType?: AiRuntimeAuthType,
  onAuthEvent?: (event: AiAuthProgressEvent) => void,
): Promise<AiRuntimeCatalog> {
  if (!configuredHost?.connect) {
    throw new Error("In-app AI account connections are unavailable in this renderer.");
  }
  const canonicalId = canonicalProviderId(providerId);
  publishCatalog(await configuredHost.connect(canonicalId, authType, onAuthEvent));
  return getAiRuntimeCatalog();
}

export async function disconnectAiRuntimeProvider(providerId: string): Promise<AiRuntimeCatalog> {
  if (!configuredHost?.disconnect) {
    throw new Error("In-app AI account disconnection is unavailable in this renderer.");
  }
  publishCatalog(await configuredHost.disconnect(canonicalProviderId(providerId)));
  return getAiRuntimeCatalog();
}

export async function saveAiRuntimeApiKey(
  providerId: string,
  apiKey: string,
): Promise<AiRuntimeCatalog> {
  if (!configuredHost?.saveApiKey) {
    throw new Error("Saving an AI API key is unavailable in this renderer.");
  }
  if (!apiKey.trim()) throw new Error("Enter an API key.");
  publishCatalog(await configuredHost.saveApiKey(canonicalProviderId(providerId), apiKey));
  return getAiRuntimeCatalog();
}

export function isAiRunCancelled(error: unknown): boolean {
  return error instanceof AiRunCancelledError;
}

export async function checkAiProviderStatus(
  provider: AiProvider | AiProviderId,
): Promise<AiProviderStatusResult> {
  const providerId = canonicalProviderId(typeof provider === "string" ? provider : provider.id);
  if (configuredHost?.checkStatus) return configuredHost.checkStatus(providerId);

  const account = runtimeCatalog.accounts.find((candidate) => candidate.providerId === providerId);
  if (account?.connectionState === "connected") {
    return { available: true, authenticated: true, message: null };
  }
  if (account?.connectionState === "error") {
    return {
      available: false,
      authenticated: false,
      inconclusive: true,
      message: account.connectionLabel,
    };
  }
  return {
    available: false,
    authenticated: false,
    message: account?.connectionLabel ?? "AI provider is not connected.",
  };
}

export function runAiPrompt({
  providerId,
  prompt,
  messages,
  agentMessages,
  modelId,
  onChunk,
  onAgentMessages,
  outputMode,
}: {
  providerId: string;
  prompt: string;
  messages?: AiConversationMessage[];
  agentMessages?: AiAgentHistoryMessage[];
  modelId?: string;
  onChunk?: (output: string) => void;
  onAgentMessages?: (messages: AiAgentHistoryMessage[]) => void;
  outputMode?: AiRunOutputMode;
}): AiRunController {
  if (!configuredHost) {
    return {
      done: Promise.reject(new Error("The native AI runtime is unavailable in this renderer.")),
      cancel: () => {},
    };
  }

  let canonicalId: AiProviderId;
  try {
    canonicalId = canonicalProviderId(providerId);
  } catch (error) {
    return {
      done: Promise.reject(error),
      cancel: () => {},
    };
  }

  return configuredHost.run({
    providerId: canonicalId,
    prompt,
    messages,
    agentMessages,
    modelId,
    onChunk,
    onAgentMessages,
    outputMode,
  });
}

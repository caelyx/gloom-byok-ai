import {
  AI_RUNNER_CAPABILITY_ID,
  type CapabilityOperation,
  type PluginCapability,
} from "gloomberb/capabilities";
import type { GloomPluginContext } from "gloomberb/types/plugin";
import {
  normalizeAiAgentHistory,
  type AiAgentHistoryMessage,
} from "../agent-history";
import { isAiProviderId, type AiProviderId } from "../providers";
import {
  isAiRunCancelled,
  type AiConversationMessage,
  type AiRunHost,
  type AiRuntimeAuthType,
} from "../runner";
import { loadPiAiHost } from "./native";

function op(
  handler: CapabilityOperation["handler"],
  kind: CapabilityOperation["kind"] = "read",
): CapabilityOperation {
  return { kind, rendererSafe: true, handler };
}

function stream(subscribe: CapabilityOperation["subscribe"]): CapabilityOperation {
  return { kind: "stream", rendererSafe: true, subscribe };
}

function requireString(value: unknown, name: string): string {
  if (typeof value !== "string" || value.length === 0) {
    throw new Error(`${name} is required.`);
  }
  return value;
}

function optionalString(value: unknown): string | undefined {
  return typeof value === "string" && value.length > 0 ? value : undefined;
}

function requireProviderId(value: unknown): AiProviderId {
  const providerId = requireString(value, "AI provider");
  if (!isAiProviderId(providerId)) {
    throw new Error(`Unknown AI provider: ${providerId}`);
  }
  return providerId;
}

function optionalAuthType(value: unknown): AiRuntimeAuthType | undefined {
  if (value === undefined || value === null || value === "") return undefined;
  if (value === "oauth" || value === "api_key") return value;
  throw new Error(`Unknown AI authentication method: ${String(value)}`);
}

function optionalMessages(value: unknown): AiConversationMessage[] | undefined {
  if (value === undefined || value === null) return undefined;
  if (!Array.isArray(value)) throw new Error("AI conversation messages must be an array.");
  return value.map((message, index) => {
    if (!message || typeof message !== "object") {
      throw new Error(`AI conversation message ${index + 1} is invalid.`);
    }
    const role = (message as { role?: unknown }).role;
    const content = (message as { content?: unknown }).content;
    if ((role !== "user" && role !== "assistant") || typeof content !== "string") {
      throw new Error(`AI conversation message ${index + 1} is invalid.`);
    }
    return { role, content };
  });
}

function optionalAgentMessages(value: unknown): AiAgentHistoryMessage[] | undefined {
  if (value === undefined || value === null) return undefined;
  if (!Array.isArray(value)) throw new Error("AI agent messages must be an array.");
  const messages = normalizeAiAgentHistory(value);
  if (messages.length !== value.length) {
    throw new Error("AI agent message history is invalid.");
  }
  return messages;
}

/**
 * The runtime, as the desktop view reaches it.
 *
 * Panes render in the view, which is a browser context with no Pi runtime and
 * no credential store, so every call it makes lands here in the Bun process
 * through this capability and streams its output back.
 */
export function createAiRunnerCapability(
  getHost: () => Promise<AiRunHost>,
): PluginCapability {
  const requireCatalogProvider = async (value: unknown): Promise<AiProviderId> => {
    const providerId = requireProviderId(value);
    const catalog = await (await getHost()).getCatalog?.();
    if (!catalog?.providers.some((provider) => provider.providerId === providerId)) {
      throw new Error(`Unknown AI provider: ${providerId}`);
    }
    return providerId;
  };

  return {
    id: AI_RUNNER_CAPABILITY_ID,
    kind: "ai-runner",
    name: "AI Runner",
    operations: {
      getCatalog: op(async () => (
        (await getHost()).getCatalog?.() ?? { providers: [], accounts: [], models: [] }
      )),
      connectProvider: stream(async (input: any, emit) => {
        const aiHost = await getHost();
        const providerId = await requireCatalogProvider(input.providerId);
        const authType = optionalAuthType(input.authType);
        if (!aiHost.connect) throw new Error("In-app AI sign-in is unavailable.");
        let disposed = false;
        aiHost.connect(providerId, authType, (event) => {
          if (!disposed) emit({ kind: "account-auth", event });
        }).then((catalog) => {
          if (!disposed) emit({ kind: "account-connected", catalog });
        }).catch((error) => {
          if (!disposed) emit({
            kind: "account-error",
            error: error instanceof Error ? error.message : String(error),
          });
        });
        return () => {
          disposed = true;
        };
      }),
      disconnectProvider: op(async (input: any) => {
        const aiHost = await getHost();
        const providerId = await requireCatalogProvider(input.providerId);
        if (!aiHost.disconnect) throw new Error("In-app AI account disconnection is unavailable.");
        return aiHost.disconnect(providerId);
      }, "action"),
      saveProviderApiKey: op(async (input: any) => {
        const aiHost = await getHost();
        const providerId = await requireCatalogProvider(input.providerId);
        const apiKey = requireString(input.apiKey, "API key");
        if (!aiHost.saveApiKey) throw new Error("Saving an AI API key is unavailable.");
        return aiHost.saveApiKey(providerId, apiKey);
      }, "action"),
      checkProviderStatus: op(async (input: any) => {
        const aiHost = await getHost();
        const providerId = await requireCatalogProvider(input.providerId);
        if (!aiHost.checkStatus) {
          throw new Error("AI provider status checks are unavailable.");
        }
        return aiHost.checkStatus(providerId);
      }),
      run: stream(async (input: any, emit) => {
        const aiHost = await getHost();
        const providerId = await requireCatalogProvider(input.providerId);
        const prompt = requireString(input.prompt, "AI prompt");
        const messages = optionalMessages(input.messages);
        const agentMessages = optionalAgentMessages(input.agentMessages);
        let completedAgentMessages: AiAgentHistoryMessage[] | undefined;
        const modelId = optionalString(input.modelId);
        const providerStatus = await aiHost.checkStatus?.(providerId);
        if (providerStatus && !providerStatus.authenticated) {
          throw new Error(
            providerStatus.message
              ?? `${providerId} is not connected.`,
          );
        }

        let disposed = false;
        const controller = aiHost.run({
          providerId,
          prompt,
          messages,
          agentMessages,
          modelId: modelId ?? undefined,
          outputMode: input.outputMode === "structured" || input.outputMode === "screener"
            ? input.outputMode
            : "plain",
          onChunk: (output) => {
            if (!disposed) emit({ kind: "chunk", output });
          },
          onAgentMessages: (nextMessages) => {
            completedAgentMessages = nextMessages;
          },
        });

        controller.done.then((output) => {
          if (!disposed) emit({
            kind: "done",
            output,
            ...(completedAgentMessages ? { agentMessages: completedAgentMessages } : {}),
          });
        }).catch((error) => {
          if (disposed) return;
          if (isAiRunCancelled(error)) {
            emit({ kind: "cancelled" });
            return;
          }
          emit({ kind: "error", error: error instanceof Error ? error.message : String(error) });
        });

        return () => {
          disposed = true;
          controller.cancel();
        };
      }),
    },
  };
}

export function registerAiRunnerCapability(ctx: GloomPluginContext): void {
  const dataDir = ctx.getConfig().dataDir;
  ctx.registerCapability(createAiRunnerCapability(() => loadPiAiHost(dataDir)));
}

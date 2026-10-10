import {
  AI_RUNNER_CAPABILITY_ID,
  getCapabilityStreamClient,
  type AiRunnerEvent,
} from "gloomberb/capabilities";
import type { GloomPluginContext } from "gloomberb/types/plugin";
import type { AiAgentHistoryMessage } from "../agent-history";
import {
  AiRunCancelledError,
  ensureAiRunHost,
  installAiRunHost,
  setAiRunHostLoader,
  type AiAuthProgressEvent,
  type AiRunHost,
  type AiRuntimeAuthType,
  type AiRuntimeCatalog,
} from "../runner";

const AI_STARTUP_READINESS_TIMEOUT_MS = 5_000;

function requireStreamClient() {
  const client = getCapabilityStreamClient();
  if (!client) throw new Error("The native AI runtime is unavailable in this renderer.");
  return client;
}

function invoke<T>(operationId: string, payload: unknown): Promise<T> {
  return requireStreamClient().invoke<T>(AI_RUNNER_CAPABILITY_ID, operationId, payload);
}

function connectProvider(
  providerId: string,
  authType?: AiRuntimeAuthType,
  onAuthEvent?: (event: AiAuthProgressEvent) => void,
): Promise<AiRuntimeCatalog> {
  return accountStream("connectProvider", { providerId, authType }, onAuthEvent);
}

/** An account operation streamed from the Bun process until it settles. */
function accountStream(
  operationId: string,
  payload: unknown,
  onAuthEvent?: (event: AiAuthProgressEvent) => void,
): Promise<AiRuntimeCatalog> {
  return new Promise<AiRuntimeCatalog>((resolve, reject) => {
    let settled = false;
    let unsubscribe: () => void = () => {};
    const settle = (callback: () => void) => {
      if (settled) return;
      settled = true;
      unsubscribe();
      callback();
    };
    try {
      unsubscribe = requireStreamClient().subscribe({
        capabilityId: AI_RUNNER_CAPABILITY_ID,
        operationId,
        payload,
        onEvent: (value) => {
          const event = value as AiRunnerEvent;
          if (event.kind === "account-connected") {
            settle(() => resolve(event.catalog as AiRuntimeCatalog));
          } else if (event.kind === "account-auth") {
            onAuthEvent?.(event.event as AiAuthProgressEvent);
          } else if (event.kind === "account-error") {
            settle(() => reject(new Error(event.error)));
          }
        },
        onError: (error) => settle(() => reject(error)),
      });
    } catch (error) {
      settle(() => reject(error));
    }
  });
}

/**
 * The view's half of the runtime: every call is forwarded to the Bun process,
 * which owns the credential store and the provider connections, and its output
 * streams back one chunk at a time.
 */
export function createBridgeAiRunHost(): AiRunHost {
  return {
    getCatalog() {
      return invoke("getCatalog", {});
    },
    connect(providerId, authType, onAuthEvent) {
      return connectProvider(providerId, authType, onAuthEvent);
    },
    disconnect(providerId) {
      return invoke("disconnectProvider", { providerId });
    },
    saveApiKey(providerId, apiKey) {
      return accountStream("saveProviderApiKey", { providerId, apiKey });
    },
    checkStatus(providerId) {
      return invoke("checkProviderStatus", { providerId });
    },
    run({
      providerId,
      prompt,
      messages,
      agentMessages,
      modelId,
      onChunk,
      onAgentMessages,
      outputMode,
    }) {
      let settled = false;
      let unsubscribe: () => void = () => {};
      let resolveDone: (output: string) => void = () => {};
      let rejectDone: (error: unknown) => void = () => {};

      const done = new Promise<string>((resolve, reject) => {
        resolveDone = resolve;
        rejectDone = reject;
      });

      const settle = (callback: () => void) => {
        if (settled) return;
        settled = true;
        unsubscribe();
        callback();
      };

      try {
        unsubscribe = requireStreamClient().subscribe({
          capabilityId: AI_RUNNER_CAPABILITY_ID,
          operationId: "run",
          payload: {
            providerId,
            prompt,
            messages,
            agentMessages,
            modelId,
            outputMode,
          },
          onEvent: (value) => {
            const event = value as AiRunnerEvent;
            switch (event.kind) {
              case "chunk":
                onChunk?.(event.output);
                break;
              case "done":
                settle(() => {
                  // Normalized by the runtime before it left the Bun process;
                  // the capability contract keeps only the discriminator.
                  if (event.agentMessages) {
                    onAgentMessages?.(event.agentMessages as AiAgentHistoryMessage[]);
                  }
                  resolveDone(event.output);
                });
                break;
              case "cancelled":
                settle(() => rejectDone(new AiRunCancelledError()));
                break;
              case "error":
                settle(() => rejectDone(new Error(event.error)));
                break;
              case "account-connected":
              case "account-auth":
              case "account-error":
                break;
            }
          },
          onError: (error) => settle(() => rejectDone(error)),
        });
      } catch (error) {
        settle(() => rejectDone(error));
      }

      return {
        done,
        cancel() {
          settle(() => rejectDone(new AiRunCancelledError()));
        },
      };
    },
  };
}

/**
 * The desktop view's wiring. Nothing heavy is loaded here: the bridge is a few
 * RPC calls, and the runtime it talks to lives in the Bun process, so provider
 * discovery starts right away rather than waiting for a pane.
 */
export function setupBridgeAiRunHost(ctx: GloomPluginContext): void {
  setAiRunHostLoader(() => installAiRunHost(createBridgeAiRunHost(), {
    catalogTimeoutMs: AI_STARTUP_READINESS_TIMEOUT_MS,
    timeoutMessage: "In-app AI provider discovery timed out during desktop startup",
    onCatalogError(error) {
      ctx.log.warn("In-app AI provider discovery could not finish during desktop startup", {
        error: error instanceof Error ? error.message : String(error),
      });
    },
  }).catch((error) => {
    ctx.log.warn("In-app AI providers could not be initialized", {
      error: error instanceof Error ? error.message : String(error),
    });
    return null;
  }));
  void ensureAiRunHost();
}

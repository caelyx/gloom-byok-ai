import type { CliCommandDef } from "gloomberb/types/plugin";
import {
  isAiProviderId,
  migrateLegacyAiProviderId,
  type AiProviderId,
} from "./providers";
import type { PiCatalog, PiTextRunController } from "./pi/runtime";

interface HeadlessPiRuntime {
  getCatalog(): Promise<PiCatalog>;
  runText(options: {
    providerId: AiProviderId;
    modelId?: string;
    prompt: string;
  }): PiTextRunController;
}

interface CreateAiCliCommandOptions {
  createRuntime?: (dataDir: string) => HeadlessPiRuntime | Promise<HeadlessPiRuntime>;
}

// The Pi runtime drags in every provider SDK. Loading it here keeps those out
// of the startup path of the TUI, which reads this command table to build its
// help before anything runs.
async function createPiRuntime(dataDir: string): Promise<HeadlessPiRuntime> {
  const { PiAiRuntime } = await import("./pi/runtime");
  return new PiAiRuntime({ dataDir });
}

/** `--flag value` and `--flag=value`, removed from `args` as they are read. */
function takeOption(args: string[], name: string): string | undefined {
  const equalsPrefix = `${name}=`;
  const equalsIndex = args.findIndex((arg) => arg.startsWith(equalsPrefix));
  if (equalsIndex >= 0) {
    const [value] = args.splice(equalsIndex, 1);
    return value!.slice(equalsPrefix.length);
  }

  const index = args.indexOf(name);
  if (index < 0) return undefined;
  const value = args[index + 1];
  args.splice(index, value == null ? 1 : 2);
  return value;
}

function configuredAiSelection(config: Record<string, unknown> | undefined): {
  providerId: AiProviderId | null;
  modelId: string | null;
} {
  const rawProviderId = typeof config?.defaultProviderId === "string"
    ? config.defaultProviderId.trim()
    : "";
  const providerId = migrateLegacyAiProviderId(rawProviderId);
  const modelId = typeof config?.defaultModelId === "string"
    ? config.defaultModelId.trim()
    : "";
  return {
    providerId: isAiProviderId(providerId) ? providerId : null,
    modelId: modelId || null,
  };
}

function connectionFailure(provider: PiCatalog["providers"][number]): string {
  if (provider.connection.state === "error") {
    return `${provider.label} could not connect: ${provider.connection.message}`;
  }
  return `${provider.label} is not connected. Sign in or enter an API key in AI pane settings, or set its API-key environment variable.`;
}

export function createAiCliCommand(options: CreateAiCliCommandOptions = {}): CliCommandDef {
  const createRuntime = options.createRuntime ?? createPiRuntime;
  return {
    name: "ai",
    description: "Inspect AI providers and run guarded headless AI prompts",
    help: {
      usage: [
        "ai providers",
        "ai ask [--provider id] [--model id] <prompt>",
      ],
    },
    execute: async (args, ctx) => {
      const action = args[0] ?? "providers";
      if (action !== "providers" && action !== "ask") {
        ctx.fail("Usage: gloomberb ai providers|ask");
      }

      const rawArgs = args.slice(1);
      const requestedProvider = takeOption(rawArgs, "--provider")?.trim();
      const requestedModel = takeOption(rawArgs, "--model")?.trim();
      const prompt = rawArgs.join(" ").trim();
      if (action === "ask" && !prompt) {
        ctx.fail("Usage: gloomberb ai ask [--provider id] [--model id] <prompt>");
      }

      const context = await ctx.initConfigData();
      try {
        const runtime = await createRuntime(context.dataDir);
        const catalog = await runtime.getCatalog();
        if (action === "providers") {
          ctx.printResult({
            data: catalog.providers.map((provider) => ({
              id: provider.id,
              name: provider.label,
              connectionState: provider.connection.state,
              connectionSource: provider.connection.state === "connected"
                ? provider.connection.source ?? ""
                : "",
              connectionError: provider.connection.state === "error"
                ? provider.connection.message
                : "",
              availableModels: provider.models.filter((model) => model.available).length,
            })),
          });
          return;
        }

        const configured = configuredAiSelection(context.config.pluginConfig.ai);
        const canonicalRequestedId = requestedProvider
          ? migrateLegacyAiProviderId(requestedProvider)
          : null;
        const requestedProviderId = canonicalRequestedId && isAiProviderId(canonicalRequestedId)
          ? canonicalRequestedId
          : null;
        const requested = requestedProviderId
          ? catalog.providers.find((provider) => provider.id === requestedProviderId)
          : null;
        if (requestedProvider && (!requestedProviderId || !requested)) {
          ctx.fail(`Unknown AI provider: ${requestedProvider}.`);
        }

        const configuredProvider = configured.providerId
          ? catalog.providers.find((provider) => (
            provider.id === configured.providerId
            && provider.connection.state === "connected"
          ))
          : null;
        const selected = requested
          ?? configuredProvider
          ?? catalog.providers.find((provider) => provider.connection.state === "connected");
        if (!selected) {
          ctx.fail("No AI provider is connected. Connect an account from AI pane settings first.");
          throw new Error("AI provider selection failed.");
        }
        if (selected.connection.state !== "connected") {
          ctx.fail(connectionFailure(selected));
        }

        const modelId = requestedModel
          || (selected.id === configured.providerId ? configured.modelId : null)
          || undefined;
        const text = await runtime.runText({
          providerId: selected.id,
          modelId,
          prompt,
        }).done;
        ctx.printResult({
          data: {
            provider: selected.id,
            model: modelId ?? null,
            text,
          },
        });
      } finally {
        context.persistence.close();
      }
    },
  };
}

export const aiCliCommand = createAiCliCommand();

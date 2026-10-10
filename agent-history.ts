export interface AiAgentTextContent {
  type: "text";
  text: string;
  textSignature?: string;
}

export interface AiAgentThinkingContent {
  type: "thinking";
  thinking: string;
  thinkingSignature?: string;
  redacted?: boolean;
}

export interface AiAgentToolCallContent {
  type: "toolCall";
  id: string;
  name: string;
  arguments: Record<string, unknown>;
  thoughtSignature?: string;
}

export type AiAgentAssistantContent =
  | AiAgentTextContent
  | AiAgentThinkingContent
  | AiAgentToolCallContent;

/**
 * Which model wrote an assistant turn. Thinking signatures are valid only for
 * that model, and Pi decides whether to replay or rewrite them by comparing
 * these with the model a thread runs on next. Turns saved before this was
 * recorded have none of them.
 */
export interface AiAgentAssistantOrigin {
  api?: string;
  provider?: string;
  model?: string;
  /** The effort Pi ran the turn at, which it re-sends for models it manages. */
  providerThinkingLevel?: string;
}

const ASSISTANT_ORIGIN_KEYS = ["api", "provider", "model", "providerThinkingLevel"] as const;

export type AiAgentHistoryMessage =
  | {
      role: "user";
      content: string;
    }
  | ({
      role: "assistant";
      content: AiAgentAssistantContent[];
    } & AiAgentAssistantOrigin)
  | {
      role: "toolResult";
      toolCallId: string;
      toolName: string;
      content: AiAgentTextContent[];
      isError: boolean;
    };

function isRecord(value: unknown): value is Record<string, unknown> {
  return !!value && typeof value === "object" && !Array.isArray(value);
}

function optionalString(value: unknown): string | undefined {
  return typeof value === "string" ? value : undefined;
}

function normalizeAssistantContent(value: unknown): AiAgentAssistantContent | null {
  if (!isRecord(value)) return null;
  if (value.type === "text" && typeof value.text === "string") {
    const textSignature = optionalString(value.textSignature);
    return {
      type: "text",
      text: value.text,
      ...(textSignature ? { textSignature } : {}),
    };
  }
  if (value.type === "thinking" && typeof value.thinking === "string") {
    const thinkingSignature = optionalString(value.thinkingSignature);
    return {
      type: "thinking",
      thinking: value.thinking,
      ...(thinkingSignature
        ? { thinkingSignature }
        : {}),
      ...(typeof value.redacted === "boolean" ? { redacted: value.redacted } : {}),
    };
  }
  if (
    value.type === "toolCall"
    && typeof value.id === "string"
    && typeof value.name === "string"
    && isRecord(value.arguments)
  ) {
    const thoughtSignature = optionalString(value.thoughtSignature);
    return {
      type: "toolCall",
      id: value.id,
      name: value.name,
      arguments: { ...value.arguments },
      ...(thoughtSignature
        ? { thoughtSignature }
        : {}),
    };
  }
  return null;
}

function normalizeToolResultContent(value: unknown): AiAgentTextContent | null {
  if (!isRecord(value) || value.type !== "text" || typeof value.text !== "string") {
    return null;
  }
  const textSignature = optionalString(value.textSignature);
  return {
    type: "text",
    text: value.text,
    ...(textSignature ? { textSignature } : {}),
  };
}

function normalizeAiAgentHistoryMessage(value: unknown): AiAgentHistoryMessage | null {
  if (!isRecord(value)) return null;
  if (value.role === "user" && typeof value.content === "string") {
    return { role: "user", content: value.content };
  }
  if (value.role === "assistant" && Array.isArray(value.content)) {
    const content = value.content.map(normalizeAssistantContent);
    if (content.some((item) => item === null)) return null;
    const origin: AiAgentAssistantOrigin = {};
    for (const key of ASSISTANT_ORIGIN_KEYS) {
      const field = optionalString(value[key]);
      if (field) origin[key] = field;
    }
    return {
      role: "assistant",
      content: content as AiAgentAssistantContent[],
      ...origin,
    };
  }
  if (
    value.role === "toolResult"
    && typeof value.toolCallId === "string"
    && typeof value.toolName === "string"
    && Array.isArray(value.content)
    && typeof value.isError === "boolean"
  ) {
    const content = value.content.map(normalizeToolResultContent);
    if (content.some((item) => item === null)) return null;
    return {
      role: "toolResult",
      toolCallId: value.toolCallId,
      toolName: value.toolName,
      content: content as AiAgentTextContent[],
      isError: value.isError,
    };
  }
  return null;
}

export function normalizeAiAgentHistory(value: unknown): AiAgentHistoryMessage[] {
  if (!Array.isArray(value)) return [];
  return value
    .map(normalizeAiAgentHistoryMessage)
    .filter((message): message is AiAgentHistoryMessage => message !== null);
}

import { ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings } from "@langchain/google-genai";
import { ChatOllama, OllamaEmbeddings } from "@langchain/ollama";
import type { BaseChatModel } from "@langchain/core/language_models/chat_models";
import type { EmbeddingsInterface } from "@langchain/core/embeddings";
import type { BaseMessage } from "@langchain/core/messages";
import { HumanMessage } from "@langchain/core/messages";
import { z } from "zod";
import { env, required } from "../config/env.js";

export type ChatModelOptions = { temperature?: number; model?: string };

/**
 * Provider-agnostic factory: every module calls `createChatModel`/`createEmbeddings`
 * without knowing whether Gemini (cloud) or Ollama (local, e.g. Qwen) is behind it.
 * Both `ChatGoogleGenerativeAI` and `ChatOllama` implement the same LangChain
 * `BaseChatModel` interface (`.invoke()`, `.pipe()`, `.withStructuredOutput()`),
 * so no call site needs to change when switching `LLM_PROVIDER`.
 */
export function createChatModel(options?: ChatModelOptions): BaseChatModel {
  if (env.llmProvider === "ollama") {
    return new ChatOllama({
      baseUrl: env.ollamaBaseUrl,
      model: options?.model ?? env.ollamaChatModel,
      temperature: options?.temperature ?? 0.7,
    });
  }

  return new ChatGoogleGenerativeAI({
    apiKey: required("GOOGLE_API_KEY", env.googleApiKey),
    model: options?.model ?? env.chatModel,
    temperature: options?.temperature ?? 0.7,
  });
}

function describeField(value: z.ZodTypeAny): string {
  const desc = value.description ? ` // ${value.description}` : "";
  const inner = value instanceof z.ZodOptional || value instanceof z.ZodNullable ? value.unwrap() : value;
  if (inner instanceof z.ZodEnum) {
    return ` — 반드시 다음 값 중 하나: ${inner.options.map((o: string) => `"${o}"`).join(" | ")}${desc}`;
  }
  if (inner instanceof z.ZodArray) {
    return ` — 문자열 배열${desc}`;
  }
  return desc;
}

function zodSchemaToJsonHint(schema: z.ZodObject<z.ZodRawShape>): string {
  const fields = Object.entries(schema.shape).map(([key, value]) => {
    return `  "${key}": ...${describeField(value as z.ZodTypeAny)}`;
  });
  return `다른 설명 없이 아래 형식의 JSON 하나만 답하세요 (필드 설명을 그대로 따르세요):\n{\n${fields.join(",\n")}\n}`;
}

/**
 * JSON-shaped generation, provider-aware:
 * - Gemini: `.withStructuredOutput(zodSchema)` — native structured output, schema-enforced.
 * - Ollama/Qwen: `withStructuredOutput`'s tool-calling/jsonSchema-grammar path was
 *   unreliable in testing on `qwen2.5:3b` (it produced tool-call-shaped garbage
 *   instead of the requested fields). Ollama's plain `format: "json"` mode
 *   (valid-JSON-only, unconstrained) worked reliably in testing, so for Ollama
 *   this appends a field-by-field hint (built from each schema field's
 *   `.describe()`) to the prompt and parses/validates the result with zod.
 */
export async function generateStructured<T extends z.ZodRawShape>(
  model: BaseChatModel,
  schema: z.ZodObject<T>,
  messages: BaseMessage[]
): Promise<z.infer<z.ZodObject<T>>> {
  if (env.llmProvider === "ollama") {
    const jsonModel = (model as ChatOllama).bind({ format: "json" });
    const response = await jsonModel.invoke([...messages, new HumanMessage(zodSchemaToJsonHint(schema))]);
    return schema.parse(JSON.parse(response.content as string));
  }

  return model.withStructuredOutput<z.infer<z.ZodObject<T>>>(schema).invoke(messages);
}

export function createEmbeddings(): EmbeddingsInterface {
  if (env.llmProvider === "ollama") {
    return new OllamaEmbeddings({
      baseUrl: env.ollamaBaseUrl,
      model: env.ollamaEmbeddingModel,
    });
  }

  return new GoogleGenerativeAIEmbeddings({
    apiKey: required("GOOGLE_API_KEY", env.googleApiKey),
    model: env.embeddingModel,
  });
}

import "dotenv/config";

function required(name: string, value: string | undefined = process.env[name]): string {
  if (!value) {
    throw new Error(`Missing required environment variable: ${name}`);
  }
  return value;
}

export type LlmProvider = "gemini" | "ollama";

const llmProvider = (process.env.LLM_PROVIDER ?? "gemini") as LlmProvider;

export const env = {
  llmProvider,
  // Only required when actually used (checked lazily in llm/provider.ts) so an
  // Ollama-only setup doesn't need a Google API key at all.
  googleApiKey: process.env.GOOGLE_API_KEY,
  chatModel: process.env.GEMINI_CHAT_MODEL ?? "gemini-flash-latest",
  embeddingModel: process.env.GEMINI_EMBEDDING_MODEL ?? "gemini-embedding-001",
  // 127.0.0.1, not localhost — Node's fetch resolves "localhost" to ::1 first,
  // which fails to connect since Ollama only binds IPv4 by default.
  ollamaBaseUrl: process.env.OLLAMA_BASE_URL ?? "http://127.0.0.1:11434",
  ollamaChatModel: process.env.OLLAMA_CHAT_MODEL ?? "qwen2.5:3b",
  ollamaEmbeddingModel: process.env.OLLAMA_EMBEDDING_MODEL ?? "nomic-embed-text",
  sessionSecret: process.env.SESSION_SECRET ?? "dev-secret",
  port: Number(process.env.PORT ?? 3000),
};

export { required };

import { createEmbeddings } from "../../llm/provider.js";

function cosineSimilarity(a: number[], b: number[]): number {
  let dot = 0;
  let normA = 0;
  let normB = 0;
  for (let i = 0; i < a.length; i++) {
    dot += a[i] * b[i];
    normA += a[i] * a[i];
    normB += b[i] * b[i];
  }
  return dot / (Math.sqrt(normA) * Math.sqrt(normB));
}

/** Standalone demo: embeds two sentences and reports their cosine similarity (no vector store). */
export async function compareSentenceSimilarity(a: string, b: string): Promise<number> {
  const embeddings = createEmbeddings();
  const [vecA, vecB] = await embeddings.embedDocuments([a, b]);
  return cosineSimilarity(vecA, vecB);
}

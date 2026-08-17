import { Document } from "@langchain/core/documents";
import { VectorStore } from "../../vectorstore/store.js";

const NAMESPACE = "03-single-pdf-rag";

export function openStore(): Promise<VectorStore> {
  return VectorStore.open(NAMESPACE);
}

/** Single-PDF module keeps only one document at a time — saving replaces whatever was there before. */
export async function saveChunks(chunks: Document[]): Promise<VectorStore> {
  const store = await openStore();
  await store.deleteAll();
  await store.addDocuments(chunks);
  return store;
}

export async function searchSimilar(store: VectorStore, query: string, k = 3): Promise<Document[]> {
  return store.similaritySearch(query, k);
}

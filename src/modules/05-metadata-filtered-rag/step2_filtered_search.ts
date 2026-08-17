import { VectorStore } from "../../vectorstore/store.js";
import { answerFromDocs, RagAnswer } from "../../vectorstore/ragChain.js";

export type MetadataFilter = Partial<{
  category: string;
  department: string;
  year: string;
  language: string;
  priority: string;
}>;

function buildFilter(filter: MetadataFilter): ((metadata: Record<string, unknown>) => boolean) | undefined {
  const entries = Object.entries(filter).filter(([, v]) => v);
  if (entries.length === 0) return undefined;
  return (metadata) => entries.every(([key, value]) => metadata[key] === value);
}

export async function searchWithFilter(
  store: VectorStore,
  query: string,
  filter: MetadataFilter,
  k = 4
): Promise<Awaited<ReturnType<VectorStore["similaritySearch"]>>> {
  return store.similaritySearch(query, k, buildFilter(filter));
}

export async function chatWithFilter(
  store: VectorStore,
  question: string,
  filter: MetadataFilter,
  k = 6
): Promise<RagAnswer> {
  return answerFromDocs(store, question, k, buildFilter(filter));
}

/** Scans stored chunk metadata to populate filter dropdown options dynamically. */
export function getAvailableMetadata(store: VectorStore): {
  categories: string[];
  departments: string[];
  years: string[];
} {
  const categories = new Set<string>();
  const departments = new Set<string>();
  const years = new Set<string>();
  for (const c of store.getAllChunks()) {
    if (c.metadata.category) categories.add(c.metadata.category as string);
    if (c.metadata.department) departments.add(c.metadata.department as string);
    if (c.metadata.year) years.add(c.metadata.year as string);
  }
  return { categories: [...categories], departments: [...departments], years: [...years] };
}

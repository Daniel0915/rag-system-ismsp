import { Document } from "@langchain/core/documents";

export type FormattedSource = { sourceFile: string; page?: number; preview: string };

/** Dedupes context docs by (source_file, page), matching the Python `format_sources()`. */
export function formatSources(docs: Document[]): FormattedSource[] {
  const seen = new Set<string>();
  const out: FormattedSource[] = [];
  for (const d of docs) {
    const key = `${d.metadata.source_file}:${d.metadata.page}`;
    if (seen.has(key)) continue;
    seen.add(key);
    out.push({
      sourceFile: d.metadata.source_file,
      page: d.metadata.page,
      preview: d.pageContent.slice(0, 200),
    });
  }
  return out;
}

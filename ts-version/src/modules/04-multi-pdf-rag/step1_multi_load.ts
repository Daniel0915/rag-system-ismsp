import fs from "node:fs";
import { Document } from "@langchain/core/documents";
import { RecursiveCharacterTextSplitter } from "@langchain/textsplitters";
import { PDFLoader } from "@langchain/community/document_loaders/fs/pdf";
import { VectorStore } from "../../vectorstore/store.js";

export type IndexResult = { status: "new" | "updated" | "skipped"; filename: string; chunks: number };

async function loadAndChunk(filePath: string, filename: string, hash: string): Promise<Document[]> {
  const loader = new PDFLoader(filePath, { splitPages: true });
  const pages = await loader.load();
  const splitter = new RecursiveCharacterTextSplitter({ chunkSize: 1000, chunkOverlap: 200 });
  const tagged = pages.map(
    (d, i) =>
      new Document({
        pageContent: d.pageContent,
        metadata: { ...d.metadata, source_file: filename, file_hash: hash, page: i + 1 },
      })
  );
  return splitter.splitDocuments(tagged);
}

/** Hash-based new/updated/skipped upsert — mirrors the Python originals' dedup pattern across modules 04-06. */
export async function indexPdf(store: VectorStore, filePath: string, filename: string): Promise<IndexResult> {
  const fileBuffer = fs.readFileSync(filePath);
  const hash = VectorStore.hashContent(fileBuffer);
  const existingHash = store.findHashBySource(filename);

  if (existingHash === hash) {
    return { status: "skipped", filename, chunks: 0 };
  }

  const chunks = await loadAndChunk(filePath, filename, hash);
  await store.replaceSourceFile(filename, chunks);
  return { status: existingHash ? "updated" : "new", filename, chunks: chunks.length };
}

export async function syncPdfsToVectorStore(
  store: VectorStore,
  files: { path: string; filename: string }[]
): Promise<IndexResult[]> {
  const results: IndexResult[] = [];
  for (const f of files) {
    results.push(await indexPdf(store, f.path, f.filename));
  }
  return results;
}

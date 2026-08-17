import fs from "node:fs";
import { Document } from "@langchain/core/documents";
import { RecursiveCharacterTextSplitter } from "@langchain/textsplitters";
import { VectorStore } from "../../vectorstore/store.js";
import { loadAnyDocument } from "../../loaders/loadDocument.js";

export type IndexResult = { status: "new" | "updated" | "skipped"; filename: string; chunks: number };

const MIN_FRAGMENT_LENGTH = 50;

/** Merges chunks shorter than MIN_FRAGMENT_LENGTH into the previous chunk, avoiding near-empty fragments. */
function mergeTinyFragments(chunks: Document[]): Document[] {
  const merged: Document[] = [];
  for (const chunk of chunks) {
    const prev = merged[merged.length - 1];
    if (prev && chunk.pageContent.length < MIN_FRAGMENT_LENGTH) {
      prev.pageContent += "\n" + chunk.pageContent;
    } else {
      merged.push(new Document({ pageContent: chunk.pageContent, metadata: chunk.metadata }));
    }
  }
  return merged;
}

export async function indexFile(store: VectorStore, filePath: string, filename: string): Promise<IndexResult> {
  const fileBuffer = fs.readFileSync(filePath);
  const hash = VectorStore.hashContent(fileBuffer);
  const existingHash = store.findHashBySource(filename);
  if (existingHash === hash) {
    return { status: "skipped", filename, chunks: 0 };
  }

  const docs = await loadAnyDocument(filePath, filename);
  const tagged = docs.map(
    (d) => new Document({ pageContent: d.pageContent, metadata: { ...d.metadata, file_hash: hash } })
  );

  const splitter = new RecursiveCharacterTextSplitter({
    chunkSize: 1000,
    chunkOverlap: 200,
    separators: ["\n\n", "\n", ". ", "! ", "? ", " ", ""],
  });
  const chunks = mergeTinyFragments(await splitter.splitDocuments(tagged));
  await store.replaceSourceFile(filename, chunks);
  return { status: existingHash ? "updated" : "new", filename, chunks: chunks.length };
}

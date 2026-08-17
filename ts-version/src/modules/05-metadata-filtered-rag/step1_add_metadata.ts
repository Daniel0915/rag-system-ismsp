import fs from "node:fs";
import { Document } from "@langchain/core/documents";
import { RecursiveCharacterTextSplitter } from "@langchain/textsplitters";
import { PDFLoader } from "@langchain/community/document_loaders/fs/pdf";
import { VectorStore } from "../../vectorstore/store.js";

export const CATEGORIES = ["계약서", "매뉴얼", "보고서", "정책", "기타"];
export const DEPARTMENTS = ["영업", "인사", "개발", "경영", "마케팅", "고객지원"];
export const PRIORITIES = ["높음", "중간", "낮음"];

export type DocMetadata = {
  category: string;
  department: string;
  year: string;
  language: string;
  priority: string;
};

export type IndexResult = { status: "new" | "updated" | "skipped"; filename: string; chunks: number };

/** Hashes file bytes + metadata together, so changing metadata alone (same file) counts as an update. */
export async function indexPdfWithMetadata(
  store: VectorStore,
  filePath: string,
  filename: string,
  metadata: DocMetadata
): Promise<IndexResult> {
  const fileBuffer = fs.readFileSync(filePath);
  const hash = VectorStore.hashContent(fileBuffer, metadata);
  const existingHash = store.findHashBySource(filename);
  if (existingHash === hash) {
    return { status: "skipped", filename, chunks: 0 };
  }

  const loader = new PDFLoader(filePath, { splitPages: true });
  const pages = await loader.load();
  const tagged = pages.map(
    (d, i) =>
      new Document({
        pageContent: d.pageContent,
        metadata: { ...d.metadata, ...metadata, source_file: filename, file_hash: hash, page: i + 1 },
      })
  );
  const splitter = new RecursiveCharacterTextSplitter({ chunkSize: 1000, chunkOverlap: 200 });
  const chunks = await splitter.splitDocuments(tagged);
  await store.replaceSourceFile(filename, chunks);
  return { status: existingHash ? "updated" : "new", filename, chunks: chunks.length };
}

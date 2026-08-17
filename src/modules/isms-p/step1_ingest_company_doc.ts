import fs from "node:fs";
import { Document } from "@langchain/core/documents";
import { loadAnyDocument } from "../../loaders/loadDocument.js";
import { VectorStore } from "../../vectorstore/store.js";
import { chunkPolicyDoc } from "./chunkPolicyDoc.js";

export const DOC_TYPES = ["정책서", "지침서", "매뉴얼", "절차서", "기타"];
export const DOMAINS = [
  { value: "1", label: "1. 관리체계 수립 및 운영" },
  { value: "2", label: "2. 보호대책 요구사항" },
  { value: "3", label: "3. 개인정보 처리단계별 요구사항" },
];

export type CompanyDocMetadata = { doc_type: string; domain: string; year: string };
export type IndexResult = { status: "new" | "updated" | "skipped"; filename: string; chunks: number };

/** Mirrors the 05 module's metadata-tagging pattern, but chunked with the ISMS-P 장-조-항 splitter. */
export async function indexCompanyDoc(
  store: VectorStore,
  filePath: string,
  filename: string,
  metadata: CompanyDocMetadata
): Promise<IndexResult> {
  const fileBuffer = fs.readFileSync(filePath);
  const hash = VectorStore.hashContent(fileBuffer, metadata);
  const existingHash = store.findHashBySource(filename);
  if (existingHash === hash) {
    return { status: "skipped", filename, chunks: 0 };
  }

  const docs = await loadAnyDocument(filePath, filename);
  const tagged = docs.map(
    (d) => new Document({ pageContent: d.pageContent, metadata: { ...d.metadata, ...metadata, source_file: filename, file_hash: hash } })
  );
  const chunks = await chunkPolicyDoc(tagged);
  await store.replaceSourceFile(filename, chunks);
  return { status: existingHash ? "updated" : "new", filename, chunks: chunks.length };
}

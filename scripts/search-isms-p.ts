import { VectorStore } from "../src/vectorstore/store.js";

const NAMESPACE = "isms-p";

const [query, sourceFile] = process.argv.slice(2);
if (!query) {
  console.error('사용법: npx tsx scripts/search-isms-p.ts "질문" ["source_file 파일명 (선택)"]');
  process.exit(1);
}

const store = await VectorStore.open(NAMESPACE);
// Normalize defensively — see the NFC/NFD note in ingest-isms-p.ts.
const normalizedSourceFile = sourceFile?.normalize("NFC");
const filter = normalizedSourceFile
  ? (metadata: Record<string, unknown>) => metadata.source_file === normalizedSourceFile
  : undefined;

const results = await store.similaritySearch(query, 4, filter);
if (results.length === 0) {
  console.log("결과 없음 (네임스페이스가 비어있거나 필터에 맞는 청크가 없음).");
}

for (const [i, doc] of results.entries()) {
  console.log(`\n[${i + 1}] source_file=${doc.metadata.source_file} page=${doc.metadata.page}`);
  console.log(doc.pageContent.slice(0, 300).replace(/\s+/g, " ").trim());
}

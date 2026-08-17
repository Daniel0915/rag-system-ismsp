import fs from "node:fs";
import path from "node:path";
import { RecursiveCharacterTextSplitter } from "@langchain/textsplitters";
import { loadAnyDocument } from "../src/loaders/loadDocument.js";
import { VectorStore } from "../src/vectorstore/store.js";

const SRC_DIR = "data/isms-p";
const NAMESPACE = "isms-p";
/** Only prose reference docs go through RAG chunking — the xlsx checklist is
 * already fully structured in data/isms-criteria.json, so re-embedding it as
 * flattened text would just be a noisier duplicate of that. */
const INCLUDE_EXT = new Set([".pdf"]);

// Ollama's local embed endpoint resets the connection on very large single
// batches (409 chunks from a 262-page PDF reliably broke it in testing;
// ~150 was reliably fine) — embed in small batches instead of one shot.
const EMBED_BATCH_SIZE = 40;

async function ingestFile(store: VectorStore, filePath: string, filename: string) {
  const fileBuffer = fs.readFileSync(filePath);
  const hash = VectorStore.hashContent(fileBuffer);
  const existingHash = store.findHashBySource(filename);
  if (existingHash === hash) {
    console.log(`skip (unchanged): ${filename}`);
    return;
  }

  const docs = await loadAnyDocument(filePath, filename);
  const tagged = docs.map((d) => ({ ...d, metadata: { ...d.metadata, file_hash: hash } }));
  const splitter = new RecursiveCharacterTextSplitter({ chunkSize: 1000, chunkOverlap: 200 });
  const chunks = await splitter.splitDocuments(tagged);

  if (existingHash) await store.deleteBySourceFile(filename);
  for (let i = 0; i < chunks.length; i += EMBED_BATCH_SIZE) {
    const batch = chunks.slice(i, i + EMBED_BATCH_SIZE);
    await store.addDocuments(batch);
    console.log(`  embedded ${Math.min(i + EMBED_BATCH_SIZE, chunks.length)}/${chunks.length}`);
  }
  console.log(`${existingHash ? "update" : "new"}: ${filename} -> ${chunks.length} chunks`);
}

const store = await VectorStore.open(NAMESPACE);
const rawEntries = fs.readdirSync(SRC_DIR).filter((f) => INCLUDE_EXT.has(path.extname(f).toLowerCase()));

for (const rawEntry of rawEntries) {
  // macOS (APFS/HFS+) returns Korean filenames from readdirSync in NFD (decomposed)
  // form, while text typed elsewhere (CLI args, browsers, other OSes) is NFC —
  // normalize the *tagged metadata* so source_file string-equality filters match,
  // but keep using the raw entry for the actual file path (Linux is
  // normalization-sensitive, so opening a re-normalized path there could 404).
  await ingestFile(store, path.join(SRC_DIR, rawEntry), rawEntry.normalize("NFC"));
}

console.log("\nsource files in namespace:", NAMESPACE);
console.table(store.listSourceFiles());

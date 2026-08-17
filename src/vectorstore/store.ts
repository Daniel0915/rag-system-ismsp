import fs from "node:fs";
import path from "node:path";
import crypto from "node:crypto";
import { Document } from "@langchain/core/documents";
import { HNSWLib } from "@langchain/community/vectorstores/hnswlib";
import { createEmbeddings } from "../llm/provider.js";

const DATA_ROOT = path.resolve(process.cwd(), "data", "vectorstore");

export type StoredChunk = {
  id: string;
  pageContent: string;
  metadata: Record<string, unknown>;
};

// One VectorStore instance per namespace, reused across requests within this
// process — see the note on `open()` below for why this matters.
const instances = new Map<string, VectorStore>();

/**
 * Thin wrapper around HNSWLib that keeps a JSON sidecar of every chunk
 * (mirrors the Python originals' Chroma `persist_directory`). HNSWLib has no
 * delete-by-id API, so deletes rebuild the index from the sidecar — the same
 * "clear then re-add" workaround the Python code already uses for Chroma,
 * just made explicit.
 */
export class VectorStore {
  private dir: string;
  private sidecarPath: string;
  private chunks: StoredChunk[] = [];
  private index: HNSWLib | null = null;
  /** Serializes writes to this namespace so concurrent uploads can't race on the JSON sidecar. */
  private writeQueue: Promise<unknown> = Promise.resolve();

  private constructor(namespace: string) {
    this.dir = path.join(DATA_ROOT, namespace);
    this.sidecarPath = path.join(this.dir, "chunks.json");
    fs.mkdirSync(this.dir, { recursive: true });
    if (fs.existsSync(this.sidecarPath)) {
      this.chunks = JSON.parse(fs.readFileSync(this.sidecarPath, "utf-8"));
    }
  }

  /**
   * Returns the shared in-process instance for this namespace, building the
   * HNSW index from disk only the first time. Every route previously called
   * `VectorStore.open(namespace)` fresh per request, which re-embedded every
   * chunk in the namespace on every single question — fine for the tutorial
   * modules' handful of demo chunks, but prohibitively slow/expensive once a
   * namespace holds a real document corpus (hundreds of chunks). Caching the
   * instance means only the *first* open() per process pays that cost;
   * subsequent reads reuse the already-built index.
   */
  static async open(namespace: string): Promise<VectorStore> {
    const existing = instances.get(namespace);
    if (existing) return existing;

    const store = new VectorStore(namespace);
    await store.rebuildIndex();
    instances.set(namespace, store);
    return store;
  }

  private async rebuildIndex(): Promise<void> {
    if (this.chunks.length === 0) {
      this.index = null;
      return;
    }
    const docs = this.chunks.map(
      (c) => new Document({ pageContent: c.pageContent, metadata: { ...c.metadata, _id: c.id } })
    );
    const index = new HNSWLib(createEmbeddings(), { space: "cosine" });
    await index.addDocuments(docs);
    this.index = index;
  }

  private persist(): void {
    fs.writeFileSync(this.sidecarPath, JSON.stringify(this.chunks, null, 2), "utf-8");
  }

  /** Runs `fn` after any writes already queued for this namespace, and queues it for the next one. */
  private enqueueWrite<T>(fn: () => Promise<T>): Promise<T> {
    const result = this.writeQueue.then(fn);
    // Swallow rejections here so one failed write doesn't wedge the queue for later writes;
    // the caller of enqueueWrite still sees the original rejection via `result`.
    this.writeQueue = result.catch(() => undefined);
    return result;
  }

  /** Hash used for change detection when re-uploading a file (matches the Python dedup pattern). */
  static hashContent(content: Buffer | string, extraMetadata?: Record<string, unknown>): string {
    const hash = crypto.createHash("md5");
    hash.update(content);
    if (extraMetadata) hash.update(JSON.stringify(extraMetadata));
    return hash.digest("hex");
  }

  findHashBySource(sourceFile: string): string | undefined {
    return this.chunks.find((c) => c.metadata.source_file === sourceFile)?.metadata
      .file_hash as string | undefined;
  }

  listSourceFiles(): { sourceFile: string; chunkCount: number; hash: string }[] {
    const bySource = new Map<string, { chunkCount: number; hash: string }>();
    for (const c of this.chunks) {
      const sourceFile = c.metadata.source_file as string | undefined;
      if (!sourceFile) continue;
      const entry = bySource.get(sourceFile) ?? { chunkCount: 0, hash: c.metadata.file_hash as string };
      entry.chunkCount += 1;
      bySource.set(sourceFile, entry);
    }
    return Array.from(bySource.entries()).map(([sourceFile, v]) => ({ sourceFile, ...v }));
  }

  async deleteBySourceFile(sourceFile: string): Promise<void> {
    return this.enqueueWrite(async () => {
      this.chunks = this.chunks.filter((c) => c.metadata.source_file !== sourceFile);
      this.persist();
      // HNSWLib has no delete-by-id, so a real deletion still needs a full rebuild —
      // but now only deletes pay that cost, not every read.
      await this.rebuildIndex();
    });
  }

  async deleteAll(): Promise<void> {
    return this.enqueueWrite(async () => {
      this.chunks = [];
      this.persist();
      this.index = null;
    });
  }

  async addDocuments(docs: Document[]): Promise<void> {
    return this.enqueueWrite(async () => {
      const newChunks: StoredChunk[] = docs.map((d) => ({
        id: crypto.randomUUID(),
        pageContent: d.pageContent,
        metadata: d.metadata,
      }));
      this.chunks.push(...newChunks);
      this.persist();

      const newDocs = newChunks.map(
        (c) => new Document({ pageContent: c.pageContent, metadata: { ...c.metadata, _id: c.id } })
      );
      if (this.index) {
        // Incremental: only embed the newly added chunks, not the whole namespace.
        await this.index.addDocuments(newDocs);
      } else {
        await this.rebuildIndex();
      }
    });
  }

  /** Replace all chunks belonging to a source file (used for the new/updated/skipped upsert flow). */
  async replaceSourceFile(sourceFile: string, docs: Document[]): Promise<void> {
    await this.deleteBySourceFile(sourceFile);
    await this.addDocuments(docs);
  }

  getAllChunks(): StoredChunk[] {
    return this.chunks;
  }

  async similaritySearch(
    query: string,
    k: number,
    filter?: (metadata: Record<string, unknown>) => boolean
  ): Promise<Document[]> {
    if (!this.index) return [];
    const docFilter = filter ? (doc: Document) => filter(doc.metadata) : undefined;
    // Over-fetch when filtering since HNSWLib filters candidates from the ANN result, not the full corpus.
    const fetchK = filter ? Math.max(k * 5, this.chunks.length) : k;
    const results = await this.index.similaritySearch(query, fetchK, docFilter);
    return results.slice(0, k);
  }
}

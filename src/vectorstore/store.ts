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

/**
 * Thin wrapper around HNSWLib that keeps a JSON sidecar of every chunk
 * (mirrors the Python originals' Chroma `persist_directory`). HNSWLib has no
 * delete-by-id API, so add/update/delete all rebuild the index from the
 * sidecar — the same "clear then re-add" workaround the Python code already
 * uses for Chroma, just made explicit. Fine at this project's data scale.
 */
export class VectorStore {
  private dir: string;
  private sidecarPath: string;
  private chunks: StoredChunk[] = [];
  private index: HNSWLib | null = null;

  private constructor(namespace: string) {
    this.dir = path.join(DATA_ROOT, namespace);
    this.sidecarPath = path.join(this.dir, "chunks.json");
    fs.mkdirSync(this.dir, { recursive: true });
    if (fs.existsSync(this.sidecarPath)) {
      this.chunks = JSON.parse(fs.readFileSync(this.sidecarPath, "utf-8"));
    }
  }

  static async open(namespace: string): Promise<VectorStore> {
    const store = new VectorStore(namespace);
    await store.rebuildIndex();
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
    this.chunks = this.chunks.filter((c) => c.metadata.source_file !== sourceFile);
    this.persist();
    await this.rebuildIndex();
  }

  async deleteAll(): Promise<void> {
    this.chunks = [];
    this.persist();
    this.index = null;
  }

  async addDocuments(docs: Document[]): Promise<void> {
    const newChunks: StoredChunk[] = docs.map((d) => ({
      id: crypto.randomUUID(),
      pageContent: d.pageContent,
      metadata: d.metadata,
    }));
    this.chunks.push(...newChunks);
    this.persist();
    await this.rebuildIndex();
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

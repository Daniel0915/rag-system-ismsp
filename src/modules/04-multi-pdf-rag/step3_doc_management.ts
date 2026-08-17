import { VectorStore } from "../../vectorstore/store.js";

export function getStoredFiles(store: VectorStore) {
  return store.listSourceFiles();
}

export async function deleteFileFromStore(store: VectorStore, sourceFile: string): Promise<void> {
  await store.deleteBySourceFile(sourceFile);
}

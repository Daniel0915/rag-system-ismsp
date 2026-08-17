import { Document } from "@langchain/core/documents";
import { RecursiveCharacterTextSplitter } from "@langchain/textsplitters";

export async function chunkDocuments(
  docs: Document[],
  chunkSize = 1000,
  chunkOverlap = 200
): Promise<Document[]> {
  const splitter = new RecursiveCharacterTextSplitter({ chunkSize, chunkOverlap });
  return splitter.splitDocuments(docs);
}

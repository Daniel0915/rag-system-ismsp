import path from "node:path";
import fs from "node:fs";
import { Document } from "@langchain/core/documents";
import { PDFLoader } from "@langchain/community/document_loaders/fs/pdf";
import { CSVLoader } from "@langchain/community/document_loaders/fs/csv";
import { DocxLoader } from "@langchain/community/document_loaders/fs/docx";
import { TextLoader } from "langchain/document_loaders/fs/text";
import officeParser from "officeparser";
import { loadImageAsDocument, isImageFile } from "./loadImage.js";

/**
 * Dispatches by extension, mirroring the Python `load_single_file` in
 * 06-rag-multi-format/app.py. pptx has no maintained LangChain.js loader, so
 * it goes through `officeparser` (plain text extraction) instead of the
 * Python original's `unstructured` — a practical approximation, not a 1:1 port.
 */
export async function loadAnyDocument(filePath: string, filename: string): Promise<Document[]> {
  const ext = path.extname(filename).toLowerCase();

  if (ext === ".pdf") {
    const loader = new PDFLoader(filePath, { splitPages: true });
    const docs = await loader.load();
    return docs.map(
      (d, i) =>
        new Document({
          pageContent: d.pageContent,
          metadata: { ...d.metadata, source_file: filename, file_type: "pdf", page: i + 1 },
        })
    );
  }

  if (ext === ".csv") {
    const loader = new CSVLoader(filePath);
    const docs = await loader.load();
    return docs.map(
      (d) => new Document({ pageContent: d.pageContent, metadata: { ...d.metadata, source_file: filename, file_type: "csv" } })
    );
  }

  if (ext === ".docx" || ext === ".doc") {
    const loader = new DocxLoader(filePath);
    const docs = await loader.load();
    return docs.map(
      (d) => new Document({ pageContent: d.pageContent, metadata: { ...d.metadata, source_file: filename, file_type: "docx" } })
    );
  }

  if (ext === ".pptx" || ext === ".ppt") {
    const text = await officeParser.parseOfficeAsync(filePath);
    return [new Document({ pageContent: text, metadata: { source_file: filename, file_type: "pptx" } })];
  }

  if (isImageFile(filename)) {
    const doc = await loadImageAsDocument(filePath, filename);
    return [doc];
  }

  // txt / md / fallback
  const loader = new TextLoader(filePath);
  const docs = await loader.load();
  return docs.map(
    (d) => new Document({ pageContent: d.pageContent, metadata: { ...d.metadata, source_file: filename, file_type: ext.replace(".", "") || "text" } })
  );
}

export function readFileBuffer(filePath: string): Buffer {
  return fs.readFileSync(filePath);
}

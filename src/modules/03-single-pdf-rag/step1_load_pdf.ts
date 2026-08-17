import { Document } from "@langchain/core/documents";
import { PDFLoader } from "@langchain/community/document_loaders/fs/pdf";

/** Loads a PDF, one Document per page — mirrors PyMuPDFLoader's page-level output. */
export async function loadPdf(filePath: string, filename: string): Promise<Document[]> {
  const loader = new PDFLoader(filePath, { splitPages: true });
  const docs = await loader.load();
  return docs.map(
    (d, i) =>
      new Document({
        pageContent: d.pageContent,
        metadata: { ...d.metadata, file_path: filePath, filename, page: i + 1 },
      })
  );
}

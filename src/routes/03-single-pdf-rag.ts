import path from "node:path";
import { Router } from "express";
import { createUpload } from "./uploadMiddleware.js";
import { loadPdf } from "../modules/03-single-pdf-rag/step1_load_pdf.js";
import { chunkDocuments } from "../modules/03-single-pdf-rag/step2_chunking.js";
import { compareSentenceSimilarity } from "../modules/03-single-pdf-rag/step3_embedding.js";
import { openStore, saveChunks } from "../modules/03-single-pdf-rag/step4_vector_store.js";
import { answerFromDocs } from "../modules/03-single-pdf-rag/step5_rag_chain.js";
import { renderPdfPageToPng } from "../pdf/renderPage.js";

const NAMESPACE = "03-single-pdf-rag";
const router = Router();
const upload = createUpload(NAMESPACE);

router.post("/upload", upload.single("file"), async (req, res) => {
  try {
    if (!req.file) return res.status(400).json({ error: "파일이 없습니다." });
    const docs = await loadPdf(req.file.path, req.file.originalname);
    const chunks = await chunkDocuments(docs);
    await saveChunks(chunks);
    res.json({
      filename: req.file.originalname,
      pages: docs.length,
      chunks: chunks.length,
      sizeBytes: req.file.size,
    });
  } catch (err) {
    res.status(500).json({ error: (err as Error).message });
  }
});

router.post("/similarity", async (req, res) => {
  try {
    const { a, b } = req.body as { a: string; b: string };
    res.json({ similarity: await compareSentenceSimilarity(a, b) });
  } catch (err) {
    res.status(500).json({ error: (err as Error).message });
  }
});

router.post("/chat", async (req, res) => {
  try {
    const { question } = req.body as { question: string };
    const store = await openStore();
    const { answer, context } = await answerFromDocs(store, question);
    res.json({
      answer,
      sources: context.map((d, i) => ({
        index: i + 1,
        page: d.metadata.page,
        filename: d.metadata.filename,
        preview: d.pageContent.slice(0, 200),
      })),
    });
  } catch (err) {
    res.status(500).json({ error: (err as Error).message });
  }
});

router.get("/chunks", async (_req, res) => {
  const store = await openStore();
  res.json({ chunks: store.getAllChunks() });
});

router.post("/delete-all", async (_req, res) => {
  const store = await openStore();
  await store.deleteAll();
  res.json({ ok: true });
});

router.get("/page-image", async (req, res) => {
  try {
    const { filename, page } = req.query as { filename: string; page: string };
    const filePath = path.join(process.cwd(), "data", "uploads", NAMESPACE, filename);
    const png = renderPdfPageToPng(filePath, Number(page));
    res.setHeader("Content-Type", "image/png");
    res.send(png);
  } catch (err) {
    res.status(500).json({ error: (err as Error).message });
  }
});

export default router;

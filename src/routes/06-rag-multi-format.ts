import path from "node:path";
import { Router } from "express";
import { createUpload } from "./uploadMiddleware.js";
import { VectorStore } from "../vectorstore/store.js";
import { indexFile } from "../modules/06-rag-multi-format/step3_unified_pipeline.js";
import { answerFromDocs } from "../vectorstore/ragChain.js";
import { formatSources } from "../modules/04-multi-pdf-rag/step2_source_display.js";
import { renderPdfPageToPng } from "../pdf/renderPage.js";

const NAMESPACE = "06-rag-multi-format";
const router = Router();
const upload = createUpload(NAMESPACE);

router.post("/upload", upload.array("files"), async (req, res) => {
  try {
    const files = (req.files as Express.Multer.File[] | undefined) ?? [];
    if (files.length === 0) return res.status(400).json({ error: "파일이 없습니다." });
    const store = await VectorStore.open(NAMESPACE);
    const results = [];
    for (const f of files) {
      results.push(await indexFile(store, f.path, f.originalname));
    }
    res.json({ results });
  } catch (err) {
    res.status(500).json({ error: (err as Error).message });
  }
});

router.post("/chat", async (req, res) => {
  try {
    const { question } = req.body as { question: string };
    const store = await VectorStore.open(NAMESPACE);
    const { answer, context } = await answerFromDocs(store, question, 5);
    res.json({
      answer,
      sources: formatSources(context).map((s, i) => ({
        ...s,
        fileType: context[i]?.metadata.file_type,
      })),
    });
  } catch (err) {
    res.status(500).json({ error: (err as Error).message });
  }
});

router.get("/files", async (_req, res) => {
  const store = await VectorStore.open(NAMESPACE);
  res.json({ files: store.listSourceFiles() });
});

router.post("/delete-file", async (req, res) => {
  const { sourceFile } = req.body as { sourceFile: string };
  const store = await VectorStore.open(NAMESPACE);
  await store.deleteBySourceFile(sourceFile);
  res.json({ ok: true });
});

router.post("/delete-all", async (_req, res) => {
  const store = await VectorStore.open(NAMESPACE);
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

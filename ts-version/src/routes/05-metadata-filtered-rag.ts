import path from "node:path";
import { Router } from "express";
import { createUpload } from "./uploadMiddleware.js";
import { VectorStore } from "../vectorstore/store.js";
import {
  indexPdfWithMetadata,
  CATEGORIES,
  DEPARTMENTS,
  PRIORITIES,
} from "../modules/05-metadata-filtered-rag/step1_add_metadata.js";
import {
  chatWithFilter,
  getAvailableMetadata,
} from "../modules/05-metadata-filtered-rag/step2_filtered_search.js";
import { formatSources } from "../modules/04-multi-pdf-rag/step2_source_display.js";
import { renderPdfPageToPng } from "../pdf/renderPage.js";

const NAMESPACE = "05-metadata-filtered-rag";
const router = Router();
const upload = createUpload(NAMESPACE);

router.get("/metadata-options", (_req, res) => {
  res.json({ categories: CATEGORIES, departments: DEPARTMENTS, priorities: PRIORITIES });
});

router.post("/upload", upload.single("file"), async (req, res) => {
  try {
    if (!req.file) return res.status(400).json({ error: "파일이 없습니다." });
    const { category, department, year, language, priority } = req.body;
    const store = await VectorStore.open(NAMESPACE);
    const result = await indexPdfWithMetadata(store, req.file.path, req.file.originalname, {
      category,
      department,
      year,
      language,
      priority,
    });
    res.json(result);
  } catch (err) {
    res.status(500).json({ error: (err as Error).message });
  }
});

router.get("/available-filters", async (_req, res) => {
  const store = await VectorStore.open(NAMESPACE);
  res.json(getAvailableMetadata(store));
});

router.post("/chat", async (req, res) => {
  try {
    const { question, filter } = req.body as { question: string; filter: Record<string, string> };
    const store = await VectorStore.open(NAMESPACE);
    const { answer, context } = await chatWithFilter(store, question, filter);
    res.json({
      answer,
      sources: formatSources(context).map((s, i) => ({
        ...s,
        category: context[i]?.metadata.category,
      })),
    });
  } catch (err) {
    res.status(500).json({ error: (err as Error).message });
  }
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

import { Router } from "express";
import { createUpload } from "./uploadMiddleware.js";
import { VectorStore } from "../vectorstore/store.js";
import { indexCompanyDoc, DOC_TYPES, DOMAINS } from "../modules/isms-p/step1_ingest_company_doc.js";
import { chatWithFilter } from "../modules/isms-p/step2_chat.js";
import { formatSources } from "../modules/04-multi-pdf-rag/step2_source_display.js";

const NAMESPACE = "isms-p-company-docs";
const router = Router();
const upload = createUpload(NAMESPACE);

router.get("/metadata-options", (_req, res) => {
  res.json({ docTypes: DOC_TYPES, domains: DOMAINS });
});

router.post("/upload", upload.single("file"), async (req, res) => {
  try {
    if (!req.file) return res.status(400).json({ error: "파일이 없습니다." });
    const { doc_type, domain, year } = req.body;
    const store = await VectorStore.open(NAMESPACE);
    const result = await indexCompanyDoc(store, req.file.path, req.file.originalname, { doc_type, domain, year });
    res.json(result);
  } catch (err) {
    res.status(500).json({ error: (err as Error).message });
  }
});

router.get("/documents", async (_req, res) => {
  const store = await VectorStore.open(NAMESPACE);
  res.json(store.listSourceFiles());
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
        doc_type: context[i]?.metadata.doc_type,
        chunk_strategy: context[i]?.metadata.chunk_strategy,
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

export default router;

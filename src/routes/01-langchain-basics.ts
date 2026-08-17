import { Router } from "express";
import { chatModelInvoke } from "../modules/01-langchain-basics/step1_chat_model.js";
import { summarize } from "../modules/01-langchain-basics/step2_prompt_template.js";
import { translate, classify, LANGUAGES, DEFAULT_CATEGORIES } from "../modules/01-langchain-basics/step3_chains.js";

const router = Router();

router.post("/step1", async (req, res) => {
  try {
    const { question } = req.body as { question: string };
    res.json({ answer: await chatModelInvoke(question) });
  } catch (err) {
    res.status(500).json({ error: (err as Error).message });
  }
});

router.post("/step2", async (req, res) => {
  try {
    const { text, length } = req.body as { text: string; length: number };
    res.json({ answer: await summarize(text, length) });
  } catch (err) {
    res.status(500).json({ error: (err as Error).message });
  }
});

router.get("/step3/options", (_req, res) => {
  res.json({ languages: LANGUAGES, categories: DEFAULT_CATEGORIES });
});

router.post("/step3/translate", async (req, res) => {
  try {
    const { text, targetLanguage } = req.body as { text: string; targetLanguage: string };
    res.json({ answer: await translate(text, targetLanguage) });
  } catch (err) {
    res.status(500).json({ error: (err as Error).message });
  }
});

router.post("/step3/classify", async (req, res) => {
  try {
    const { text } = req.body as { text: string };
    res.json({ answer: await classify(text) });
  } catch (err) {
    res.status(500).json({ error: (err as Error).message });
  }
});

export default router;

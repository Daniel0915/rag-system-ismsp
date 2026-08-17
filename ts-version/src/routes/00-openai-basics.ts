import { Router } from "express";
import { askHello } from "../modules/00-openai-basics/step1_hello.js";
import { askWithRole, ROLE_OPTIONS } from "../modules/00-openai-basics/step2_role.js";
import { analyzeText } from "../modules/00-openai-basics/step3_json_output.js";

const router = Router();

router.post("/step1", async (req, res) => {
  try {
    const { question, temperature } = req.body as { question: string; temperature: number };
    const answer = await askHello(question, temperature);
    res.json({ answer });
  } catch (err) {
    res.status(500).json({ error: (err as Error).message });
  }
});

router.get("/step2/roles", (_req, res) => {
  res.json(ROLE_OPTIONS);
});

router.post("/step2", async (req, res) => {
  try {
    const { roleKey, input, customSystemPrompt } = req.body as {
      roleKey: string;
      input: string;
      customSystemPrompt?: string;
    };
    const answer = await askWithRole(roleKey, input, customSystemPrompt);
    res.json({ answer });
  } catch (err) {
    res.status(500).json({ error: (err as Error).message });
  }
});

router.post("/step3", async (req, res) => {
  try {
    const { text } = req.body as { text: string };
    const analysis = await analyzeText(text);
    res.json(analysis);
  } catch (err) {
    res.status(500).json({ error: (err as Error).message });
  }
});

export default router;

import { Router } from "express";
import { runSimpleGraph } from "../modules/07-langgraph-basics/step1_simple_graph.js";
import { runConditionalGraph } from "../modules/07-langgraph-basics/step2_conditional_branch.js";
import { runLoopGraph } from "../modules/07-langgraph-basics/step3_loop_graph.js";
import { runRagWithEvaluator } from "../modules/07-langgraph-basics/step4_rag_with_evaluator.js";

const router = Router();

router.post("/step1", async (req, res) => {
  try {
    const { question } = req.body as { question: string };
    res.json(await runSimpleGraph(question));
  } catch (err) {
    res.status(500).json({ error: (err as Error).message });
  }
});

router.post("/step2", async (req, res) => {
  try {
    const { question } = req.body as { question: string };
    res.json(await runConditionalGraph(question));
  } catch (err) {
    res.status(500).json({ error: (err as Error).message });
  }
});

router.post("/step3", async (req, res) => {
  try {
    const { topic, maxIterations } = req.body as { topic: string; maxIterations: number };
    res.json({ history: await runLoopGraph(topic, maxIterations) });
  } catch (err) {
    res.status(500).json({ error: (err as Error).message });
  }
});

router.post("/step4", async (req, res) => {
  try {
    const { question, maxIterations } = req.body as { question: string; maxIterations: number };
    res.json(await runRagWithEvaluator(question, maxIterations));
  } catch (err) {
    res.status(500).json({ error: (err as Error).message });
  }
});

export default router;

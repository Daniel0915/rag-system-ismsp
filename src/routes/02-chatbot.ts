import { Router } from "express";
import { replyToChat } from "../modules/02-chatbot/step2_chat_ui.js";
import { PERSONAS } from "../modules/02-chatbot/step3_chatbot.js";

const router = Router();
const GENERIC_SYSTEM_PROMPT = "당신은 친절한 AI 도우미입니다. 한국어로 답변하세요.";

// Step 1 — single-shot widget demo (topic/level selectors), no history.
router.post("/step1", async (req, res) => {
  try {
    const { name, topic, level } = req.body as { name: string; topic: string; level: string };
    const question = `${name}님을 위해 "${topic}" 주제를 "${level}" 난이도로 짧게 설명해주세요.`;
    const answer = await replyToChat([], GENERIC_SYSTEM_PROMPT, question);
    res.json({ answer });
  } catch (err) {
    res.status(500).json({ error: (err as Error).message });
  }
});

// Step 2 — chat UI with session-held history.
router.get("/step2/history", (req, res) => {
  res.json({ history: req.session.chatHistoryBasic ?? [] });
});

router.post("/step2/message", async (req, res) => {
  try {
    const { message } = req.body as { message: string };
    const history = req.session.chatHistoryBasic ?? [];
    const answer = await replyToChat(history, GENERIC_SYSTEM_PROMPT, message);
    history.push({ role: "user", content: message }, { role: "assistant", content: answer });
    req.session.chatHistoryBasic = history;
    res.json({ answer, history });
  } catch (err) {
    res.status(500).json({ error: (err as Error).message });
  }
});

router.post("/step2/reset", (req, res) => {
  req.session.chatHistoryBasic = [];
  res.json({ ok: true });
});

// Step 3 — persona-based chatbot; switching persona resets history.
router.get("/step3/personas", (_req, res) => {
  res.json(PERSONAS);
});

router.post("/step3/select-persona", (req, res) => {
  const { personaKey, customSystemPrompt } = req.body as { personaKey: string; customSystemPrompt?: string };
  req.session.currentPersona = personaKey;
  req.session.chatHistoryPersona = [];
  const greeting =
    personaKey === "custom"
      ? "안녕하세요! 무엇을 도와드릴까요?"
      : PERSONAS[personaKey]?.greeting ?? "";
  res.json({ greeting, customSystemPrompt: customSystemPrompt ?? "" });
});

router.post("/step3/message", async (req, res) => {
  try {
    const { message, customSystemPrompt } = req.body as { message: string; customSystemPrompt?: string };
    const personaKey = req.session.currentPersona ?? "chef";
    const systemPrompt =
      personaKey === "custom" ? customSystemPrompt ?? "" : PERSONAS[personaKey]?.systemPrompt ?? "";
    const history = req.session.chatHistoryPersona ?? [];
    const answer = await replyToChat(history, systemPrompt, message);
    history.push({ role: "user", content: message }, { role: "assistant", content: answer });
    req.session.chatHistoryPersona = history;
    res.json({ answer, history });
  } catch (err) {
    res.status(500).json({ error: (err as Error).message });
  }
});

export default router;

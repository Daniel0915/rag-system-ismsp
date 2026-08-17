import { z } from "zod";
import { HumanMessage, SystemMessage } from "@langchain/core/messages";
import { createChatModel, generateStructured } from "../../llm/provider.js";

const AnalysisSchema = z.object({
  summary: z.string().describe("텍스트 요약 (한 문장)"),
  keywords: z.array(z.string()).describe("핵심 키워드 목록"),
  sentiment: z.enum(["긍정", "부정", "중립"]).describe("전체적인 감정"),
  category: z.string().describe("텍스트의 주제 분류"),
});

export type TextAnalysis = z.infer<typeof AnalysisSchema>;

const SYSTEM_PROMPT =
  "당신은 텍스트 분석 전문가입니다. 주어진 텍스트를 분석해서 summary, keywords, sentiment, category를 채우세요.";

/**
 * Uses structured output (`generateStructured`) instead of the Python
 * original's prompt-instructed JSON + manual `JSON.parse` with a try/except
 * fallback — a straightforward upgrade since both Gemini and Ollama support
 * some form of enforced JSON output.
 */
export async function analyzeText(text: string): Promise<TextAnalysis> {
  const model = createChatModel({ temperature: 0 });
  return generateStructured(model, AnalysisSchema, [new SystemMessage(SYSTEM_PROMPT), new HumanMessage(text)]);
}

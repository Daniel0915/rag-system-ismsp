import { HumanMessage, SystemMessage } from "@langchain/core/messages";
import { createChatModel } from "../../llm/provider.js";

export const ROLE_OPTIONS: Record<string, { label: string; systemPrompt: string }> = {
  translator: { label: "번역가", systemPrompt: "당신은 전문 번역가입니다. 입력된 문장을 자연스러운 영어로 번역하세요." },
  reviewer: { label: "코드 리뷰어", systemPrompt: "당신은 꼼꼼한 코드 리뷰어입니다. 코드의 문제점과 개선점을 한국어로 지적하세요." },
  summarizer: { label: "요약 전문가", systemPrompt: "당신은 요약 전문가입니다. 입력된 텍스트를 핵심만 간결하게 한국어로 요약하세요." },
};

export async function askWithRole(
  roleKey: string,
  input: string,
  customSystemPrompt?: string
): Promise<string> {
  const systemPrompt =
    roleKey === "custom" ? customSystemPrompt ?? "" : ROLE_OPTIONS[roleKey]?.systemPrompt ?? "";
  const model = createChatModel();
  const response = await model.invoke([new SystemMessage(systemPrompt), new HumanMessage(input)]);
  return response.content as string;
}

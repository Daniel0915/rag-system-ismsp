import { HumanMessage, SystemMessage } from "@langchain/core/messages";
import { createChatModel } from "../../llm/provider.js";

const DEFAULT_SYSTEM_PROMPT = "당신은 친절한 AI 도우미입니다. 한국어로 답변하세요.";

export async function askHello(question: string, temperature: number): Promise<string> {
  const clamped = Math.min(1, Math.max(0, temperature));
  const model = createChatModel({ temperature: clamped });
  const response = await model.invoke([new SystemMessage(DEFAULT_SYSTEM_PROMPT), new HumanMessage(question)]);
  return response.content as string;
}

import { HumanMessage, SystemMessage } from "@langchain/core/messages";
import { createChatModel } from "../../llm/provider.js";

const SYSTEM_PROMPT = "당신은 친절한 AI 도우미입니다. 한국어로 답변하세요.";

export async function chatModelInvoke(question: string): Promise<string> {
  const model = createChatModel();
  const response = await model.invoke([new SystemMessage(SYSTEM_PROMPT), new HumanMessage(question)]);
  return response.content as string;
}

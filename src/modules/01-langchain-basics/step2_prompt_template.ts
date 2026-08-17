import { ChatPromptTemplate } from "@langchain/core/prompts";
import { StringOutputParser } from "@langchain/core/output_parsers";
import { createChatModel } from "../../llm/provider.js";

const prompt = ChatPromptTemplate.fromMessages([
  ["system", "당신은 요약 전문가입니다. 주어진 텍스트를 {length}줄로 요약하세요."],
  ["human", "{text}"],
]);

export async function summarize(text: string, length: number): Promise<string> {
  const chain = prompt.pipe(createChatModel()).pipe(new StringOutputParser());
  return chain.invoke({ text, length });
}

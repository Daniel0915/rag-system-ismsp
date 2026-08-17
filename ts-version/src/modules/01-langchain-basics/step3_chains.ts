import { ChatPromptTemplate } from "@langchain/core/prompts";
import { StringOutputParser } from "@langchain/core/output_parsers";
import { createChatModel } from "../../llm/provider.js";

export const LANGUAGES = ["영어", "일본어", "중국어", "프랑스어"];
export const DEFAULT_CATEGORIES = ["기술", "경제", "스포츠", "문화", "정치"];

const translatePrompt = ChatPromptTemplate.fromMessages([
  ["system", "당신은 전문 번역가입니다. 주어진 텍스트를 {target_language}로 번역하세요."],
  ["human", "{text}"],
]);

const classifyPrompt = ChatPromptTemplate.fromMessages([
  ["system", "당신은 텍스트 분류 전문가입니다. 다음 카테고리 중 하나로만 답하세요: {categories}"],
  ["human", "{text}"],
]);

export async function translate(text: string, targetLanguage: string): Promise<string> {
  const chain = translatePrompt.pipe(createChatModel({ temperature: 0.3 })).pipe(new StringOutputParser());
  return chain.invoke({ text, target_language: targetLanguage });
}

export async function classify(text: string, categories: string[] = DEFAULT_CATEGORIES): Promise<string> {
  const chain = classifyPrompt.pipe(createChatModel({ temperature: 0 })).pipe(new StringOutputParser());
  return chain.invoke({ text, categories: categories.join(", ") });
}

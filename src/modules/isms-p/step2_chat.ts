import { Document } from "@langchain/core/documents";
import { ChatPromptTemplate } from "@langchain/core/prompts";
import { StringOutputParser } from "@langchain/core/output_parsers";
import { createChatModel } from "../../llm/provider.js";
import { VectorStore } from "../../vectorstore/store.js";

export type MetadataFilter = Partial<{ doc_type: string; domain: string }>;

function buildFilter(filter: MetadataFilter): ((metadata: Record<string, unknown>) => boolean) | undefined {
  const entries = Object.entries(filter).filter(([, v]) => v);
  if (entries.length === 0) return undefined;
  return (metadata) => entries.every(([key, value]) => metadata[key] === value);
}

const PROMPT = ChatPromptTemplate.fromMessages([
  [
    "system",
    "당신은 기업의 ISMS-P(정보보호 및 개인정보보호 관리체계) 인증 준비를 돕는 어시스턴트입니다. " +
      "아래 context는 이 기업이 등록한 정책/지침 문서에서 발췌한 내용입니다. context만 근거로 답변하고, " +
      "context에 없는 내용이면 모른다고 답하세요. 답변에는 근거가 된 문서 파일명을 함께 언급하세요. " +
      "한국어로 답변하세요.\n\ncontext:\n{context}",
  ],
  ["human", "{question}"],
]);

export type ChatAnswer = { answer: string; context: Document[] };

export async function chatWithFilter(
  store: VectorStore,
  question: string,
  filter: MetadataFilter,
  k = 4
): Promise<ChatAnswer> {
  const context = await store.similaritySearch(question, k, buildFilter(filter));
  const contextText = context
    .map((d, i) => `[${i + 1}] (출처: ${d.metadata.source_file}, ${d.metadata.chunk_strategy}) ${d.pageContent}`)
    .join("\n\n");

  const chain = PROMPT.pipe(createChatModel()).pipe(new StringOutputParser());
  const answer = await chain.invoke({ context: contextText, question });
  return { answer, context };
}

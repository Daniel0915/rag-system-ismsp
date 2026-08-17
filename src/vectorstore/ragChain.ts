import { Document } from "@langchain/core/documents";
import { ChatPromptTemplate } from "@langchain/core/prompts";
import { StringOutputParser } from "@langchain/core/output_parsers";
import { createChatModel } from "../llm/provider.js";
import { VectorStore } from "./store.js";

const RAG_PROMPT = ChatPromptTemplate.fromMessages([
  [
    "system",
    "당신은 문서 기반 질의응답 도우미입니다. 아래 context만 사용해서 답변하세요. " +
      "context에 없는 내용이면 모른다고 답하세요. 답변에는 근거가 된 출처 파일명을 함께 언급하세요. " +
      "한국어로 답변하세요.\n\ncontext:\n{context}",
  ],
  ["human", "{question}"],
]);

export type RagAnswer = { answer: string; context: Document[] };

/**
 * Shared retrieve → stuff-into-prompt → generate helper used by modules
 * 03-06, replicating LangChain Python's `create_retrieval_chain` +
 * `create_stuff_documents_chain` against our custom HNSWLib-backed
 * VectorStore (which isn't a LangChain.js Runnable retriever).
 */
export async function answerFromDocs(
  store: VectorStore,
  question: string,
  k = 3,
  filter?: (metadata: Record<string, unknown>) => boolean
): Promise<RagAnswer> {
  const context = await store.similaritySearch(question, k, filter);
  const contextText = context
    .map((d, i) => `[${i + 1}] (출처: ${d.metadata.source_file ?? d.metadata.filename}) ${d.pageContent}`)
    .join("\n\n");

  const chain = RAG_PROMPT.pipe(createChatModel()).pipe(new StringOutputParser());
  const answer = await chain.invoke({ context: contextText, question });
  return { answer, context };
}

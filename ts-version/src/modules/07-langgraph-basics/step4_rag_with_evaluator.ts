import { z } from "zod";
import { StateGraph, Annotation, START, END } from "@langchain/langgraph";
import { HumanMessage } from "@langchain/core/messages";
import { Document } from "@langchain/core/documents";
import { HNSWLib } from "@langchain/community/vectorstores/hnswlib";
import { createChatModel, createEmbeddings, generateStructured } from "../../llm/provider.js";

/**
 * Synthetic 9-document corpus with a deliberate 2-hop "bridge": the NovaCore
 * processor doc names Qubrix as its maker, and a separate Qubrix doc has the
 * financials — a single k=2 retrieval on the original question won't surface
 * both, which is exactly what forces the rewrite→retrieve retry loop below.
 */
const TEST_DOCUMENTS = [
  new Document({
    pageContent: "Apple은 아이폰, 맥, 아이패드를 만드는 미국의 대표적인 빅테크 기업입니다.",
    metadata: { source: "Apple 개요" },
  }),
  new Document({
    pageContent: "Samsung은 스마트폰과 반도체를 함께 생산하는 한국의 대표적인 글로벌 전자기업입니다.",
    metadata: { source: "Samsung 개요" },
  }),
  new Document({
    pageContent: "Google은 검색엔진과 안드로이드 OS, 클라우드 서비스를 제공하는 미국 기업입니다.",
    metadata: { source: "Google 개요" },
  }),
  new Document({
    pageContent: "Microsoft는 윈도우 운영체제와 Azure 클라우드, Office 제품군으로 유명한 미국 기업입니다.",
    metadata: { source: "Microsoft 개요" },
  }),
  new Document({
    pageContent: "Qubrix는 저전력 AI 반도체를 전문으로 설계하는 팹리스 반도체 회사입니다.",
    metadata: { source: "Qubrix 개요" },
  }),
  new Document({
    pageContent: "NovaCore는 Qubrix가 설계하고 생산하는, 온디바이스 AI 연산에 특화된 프로세서입니다.",
    metadata: { source: "NovaCore 프로세서" },
  }),
  new Document({
    pageContent: "Qubrix의 지난해 매출은 4200억원이었습니다.",
    metadata: { source: "Qubrix 재무 정보" },
  }),
  new Document({
    pageContent: "Apple의 지난해 매출은 3830억 달러였습니다.",
    metadata: { source: "Apple 재무 정보" },
  }),
  new Document({
    pageContent: "Samsung의 지난해 매출은 300조원이었습니다.",
    metadata: { source: "Samsung 재무 정보" },
  }),
  new Document({
    pageContent: "아이폰15는 A17 프로 칩을 탑재했으며 티타늄 프레임을 사용합니다.",
    metadata: { source: "아이폰15 제품 정보" },
  }),
  new Document({
    pageContent: "갤럭시S24는 엑시노스 또는 스냅드래곤 칩을 지역에 따라 탑재합니다.",
    metadata: { source: "갤럭시S24 제품 정보" },
  }),
];

let cachedStore: Promise<HNSWLib> | null = null;
function getTestVectorStore(): Promise<HNSWLib> {
  if (!cachedStore) {
    cachedStore = (async () => {
      const index = new HNSWLib(createEmbeddings(), { space: "cosine" });
      await index.addDocuments(TEST_DOCUMENTS);
      return index;
    })();
  }
  return cachedStore;
}

const State = Annotation.Root({
  question: Annotation<string>(),
  searchQuery: Annotation<string>(),
  answer: Annotation<string>(),
  isComplete: Annotation<boolean>(),
  evaluationReason: Annotation<string>(),
  iteration: Annotation<number>(),
  maxIterations: Annotation<number>(),
  allContext: Annotation<string[]>({
    reducer: (a, b) => a.concat(b),
    default: () => [],
  }),
});

async function rewriteQuery(state: typeof State.State) {
  const iteration = (state.iteration ?? 0) + 1;
  const model = createChatModel({ temperature: 0 });
  const prompt =
    iteration === 1
      ? `다음 질문을 벡터 검색에 적합한 검색어로 바꾸세요. 검색어만 답하세요.\n\n질문: ${state.question}`
      : `이전 검색어로는 충분한 정보를 찾지 못했습니다. 더 나은 검색어를 만드세요. 검색어만 답하세요.\n\n` +
        `원래 질문: ${state.question}\n이전 검색어: ${state.searchQuery}\n부족했던 이유: ${state.evaluationReason}`;
  const response = await model.invoke([new HumanMessage(prompt)]);
  return { searchQuery: (response.content as string).trim(), iteration };
}

/**
 * k=1 (tighter than the Python original's k=2) — with Gemini's embedding
 * space, a k=2 first-pass query tended to surface both the bridge doc and
 * the financials doc at once, skipping the retry demo entirely. k=1 makes
 * the single-retrieval-is-insufficient behavior reliable.
 */
async function retrieve(state: typeof State.State) {
  const store = await getTestVectorStore();
  const docs = await store.similaritySearch(state.searchQuery, 1);
  const formatted = docs.map((d) => `(출처: ${d.metadata.source}) ${d.pageContent}`);
  return { allContext: formatted };
}

async function generate(state: typeof State.State) {
  const uniqueContext = Array.from(new Set(state.allContext));
  const contextBlock = uniqueContext.map((c, i) => `[${i + 1}] ${c}`).join("\n");
  const model = createChatModel({ temperature: 0 });
  const response = await model.invoke([
    new HumanMessage(
      `아래 context만 사용해서 질문에 답하세요. context에서 확인할 수 없으면 ` +
        `"제공된 문서에서 확인할 수 없습니다."라고 답하세요.\n\ncontext:\n${contextBlock}\n\n질문: ${state.question}`
    ),
  ]);
  return { answer: response.content as string };
}

const EvaluationSchema = z.object({
  isComplete: z.boolean().describe("답변이 질문에 충분히 답했는지 여부"),
  reason: z.string().describe("판단 이유"),
});

async function evaluate(state: typeof State.State) {
  const model = createChatModel({ temperature: 0 });
  const result = await generateStructured(model, EvaluationSchema, [
    new HumanMessage(`질문: ${state.question}\n\n답변: ${state.answer}\n\n이 답변이 질문에 충분히 답했나요?`),
  ]);
  return { isComplete: result.isComplete, evaluationReason: result.reason };
}

function shouldRetry(state: typeof State.State): "retry" | "done" {
  if (!state.isComplete && state.iteration < state.maxIterations) return "retry";
  return "done";
}

const graph = new StateGraph(State)
  .addNode("rewrite_query", rewriteQuery)
  .addNode("retrieve", retrieve)
  .addNode("generate", generate)
  .addNode("evaluate", evaluate)
  .addEdge(START, "rewrite_query")
  .addEdge("rewrite_query", "retrieve")
  .addEdge("retrieve", "generate")
  .addEdge("generate", "evaluate")
  .addConditionalEdges("evaluate", shouldRetry, { retry: "rewrite_query", done: END });

export const ragEvaluatorGraphApp = graph.compile();

export async function runRagWithEvaluator(question: string, maxIterations: number) {
  return ragEvaluatorGraphApp.invoke({
    question,
    searchQuery: "",
    answer: "",
    isComplete: false,
    evaluationReason: "",
    iteration: 0,
    maxIterations,
  });
}
